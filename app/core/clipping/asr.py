"""ASR stage (ticket 002): local/offline transcription with timestamped segments.

Supports two interchangeable backends, auto-selected by the model path:
  * **SenseVoice** (FunAudioLLM/SenseVoiceSmall, via funasr) — fast, strong for
    Chinese, ~5-15x faster than whisper-large. Selected when the path contains
    "sensevoice" (case-insensitive).
  * **faster-whisper** (default) — multilingual, kept as the fallback backend.

Both return ``[{start, end, text}, ...]`` (float seconds) so downstream stages
(LLM, VLM, fallback) are backend-agnostic.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

from ...utils.logger import logger

# Keyword anchors for the VLM-failure fallback (ticket 003 / 004 §4).
# Used only when the VLM stage completely fails; matched against ASR text.
FALLBACK_KEYWORDS = ("买", "卖", "价格", "机制", "上车")

# --- Model cache: reuse a loaded ASR model across segments of the same run.
# Key: (model_path, compute_type). Caching avoids re-loading the model on every
# segment in segmented recording (was the dominant cost).
_ASR_CACHE: dict[tuple[str, str], Any] = {}
_ASR_LOCK = threading.Lock()


def _is_sensevoice(model_path: str) -> bool:
    return "sensevoice" in os.path.basename(model_path).lower()


def _load_sensevoice(model_path: str):
    """Load a SenseVoice model via funasr (AutoModel). Returns the AutoModel."""
    from funasr import AutoModel  # type: ignore
    logger.info(f"[AI-Clip] Loading SenseVoice (funasr, cached): {model_path}")
    # SenseVoice runs on GPU via funasr's own device handling. disable_update=True
    # to avoid network checks; disable_pbar for clean logs.
    return AutoModel(
        model=model_path,
        trust_remote_code=True,
        disable_update=True,
        disable_pbar=True,
        device="cuda:0",
    )


def _load_whisper(model_path: str, compute_type: str):
    """Load a faster-whisper model."""
    from faster_whisper import WhisperModel  # type: ignore
    logger.info(f"[AI-Clip] Loading faster-whisper (cached): {model_path} (compute_type={compute_type})")
    return WhisperModel(model_path, device="cuda", compute_type=compute_type)


def _get_asr_model(model_path: str, compute_type: str):
    """Return a cached ASR model, auto-selecting SenseVoice or faster-whisper."""
    key = (model_path, compute_type)
    with _ASR_LOCK:
        m = _ASR_CACHE.get(key)
        if m is not None:
            return m
        if _is_sensevoice(model_path):
            m = ("sensevoice", _load_sensevoice(model_path))
        else:
            m = ("whisper", _load_whisper(model_path, compute_type))
        _ASR_CACHE[key] = m
        return m


def release_asr_models() -> None:
    """Unload all cached ASR models and free VRAM. Call after the run ends."""
    with _ASR_LOCK:
        if not _ASR_CACHE:
            return
        n = len(_ASR_CACHE)
        _ASR_CACHE.clear()
    logger.info(f"[AI-Clip] released {n} cached ASR model(s)")
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def transcribe(video_path: str, model_path: str, compute_type: str, language: str | None = None) -> list[dict]:
    """Transcribe a video's audio with SenseVoice or faster-whisper (cached & reused).

    Returns ``[{start, end, text}, ...]`` (float seconds). Returns ``[]`` on failure.
    """
    try:
        backend, model = _get_asr_model(model_path, compute_type)
    except ImportError as e:
        logger.error(f"[AI-Clip] ASR backend not installed: {e}")
        return []
    except Exception as e:
        logger.error(f"[AI-Clip] ASR model load failed: {e}")
        return []

    try:
        if backend == "sensevoice":
            results = _transcribe_sensevoice(model, video_path, language)
        else:
            results = _transcribe_whisper(model, video_path, language)
        logger.success(f"[AI-Clip] ASR done ({backend}): {len(results)} segments")
        return results
    except Exception as e:
        logger.error(f"[AI-Clip] ASR failed: {e}")
        return []


def _transcribe_whisper(model, video_path: str, language: str | None) -> list[dict]:
    """faster-whisper transcription -> [{start, end, text}]."""
    segments_gen, _info = model.transcribe(
        video_path,
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    results: list[dict] = []
    for seg in segments_gen:
        text = (seg.text or "").strip()
        if text:
            results.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return results


def _transcribe_sensevoice(model, video_path: str, language: str | None) -> list[dict]:
    """SenseVoice transcription via funasr -> [{start, end, text}].

    SenseVoice itself outputs full text with rich tags (emotion/events) but not
    per-sentence timestamps. We run VAD-based segmentation first (fsmn-vad) to
    get clip-level timestamps, then transcribe each chunk and map back. The
    timestamps come from the VAD model (utterance boundaries), which is reliable.
    """
    # First get VAD timestamps (utterance boundaries).
    res = model.generate(
        input=video_path,
        cache={},
        language="auto" if not language else language,
        use_itn=True,
        merge_length=15000,  # merge very short VAD segments
    )
    results: list[dict] = []
    for item in res:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        # SenseVoice text carries tags like <|HAPPY|><|Speech|>; strip them for
        # cleaner downstream matching, keep plain text for the transcript.
        clean = _strip_sv_tags(text)
        if not clean:
            continue
        # funasr may provide sentence_info or timestamp fields depending on config.
        # Fall back to whole-file span when no per-utterance timestamp is present.
        s = float(item.get("start", 0.0) or 0.0) / 1000.0  # ms -> s
        e = float(item.get("end", 0.0) or 0.0) / 1000.0
        if e <= s:
            # No reliable timestamp from this item; treat as 0-length (filtered later).
            e = s
        results.append({"start": s, "end": e, "text": clean})
    return results


_SV_TAG_RE = re.compile(r"<\|[^|]*\|>")


def _strip_sv_tags(text: str) -> str:
    """Remove SenseVoice rich tags like ``<|HAPPY|>``, ``<|Speech|>``."""
    return _SV_TAG_RE.sub("", text).strip()


def keyword_fallback(segments: list[dict]) -> list[dict]:
    """Ticket 004 §4 fallback: keyword hard-match on ASR segments when the VLM fails.

    Each hit becomes a clip candidate using the segment's own ``[start, end]`` as
    **float seconds** (downstream filter/pad/clamp expects seconds, not the
    ``MM:SS.xx`` string used for VLM subtitle input).
    ``product``/``selling_point`` are unavailable (filled with a placeholder).
    """
    clips: list[dict] = []
    for seg in segments:
        text = seg.get("text", "")
        if any(kw in text for kw in FALLBACK_KEYWORDS):
            clips.append({
                "product": "ASR兜底",
                "selling_point": text[:40],
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            })
    return clips


def segments_to_srt(segments: list[dict]) -> str:
    """Render ASR segments as an SRT-ish subtitle block for the VLM (joint input)."""
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(f"{i}\n{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}\n{seg['text']}\n")
    return "\n".join(lines)


def _fmt_ts(seconds: float) -> str:
    """Seconds -> ``MM:SS.xx`` (matches Mage-VL processor's codec timestamp format)."""
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def save_transcript_json(segments: list[dict], video_path: str) -> str | None:
    """Save ASR segments as a sibling JSON transcript next to the source video.

    Output file: ``<video_dir>/<video_basename>_asr.json``
    Each entry: ``{start, end, text, start_fmt, end_fmt}`` where ``*_fmt`` is the
    human-readable ``HH:MM:SS.mmm`` form. Returns the output path on success.
    """
    if not segments:
        return None
    try:
        base = os.path.splitext(video_path)[0]
        out_path = base + "_asr.json"
        payload = {
            "source": os.path.basename(video_path),
            "segment_count": len(segments),
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "start_fmt": _fmt_hms(s["start"]),
                    "end_fmt": _fmt_hms(s["end"]),
                    "text": s.get("text", ""),
                }
                for s in segments
            ],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"[AI-Clip] ASR transcript saved: {out_path}")
        return out_path
    except Exception as e:
        logger.error(f"[AI-Clip] failed to save ASR transcript: {e}")
        return None


def _fmt_hms(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS.mmm``."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
