"""ASR stage (ticket 002): faster-whisper, local/offline, serial-first.

Runs first in the ASR->VLM order (ticket 004). Produces timestamped segments
``[{start, end, text}]`` consumed as the VLM's subtitle input. On failure the
caller falls back to keyword hard-match (ticket 004 §4).
"""
from __future__ import annotations

from typing import Any

from ...utils.logger import logger

# Keyword anchors for the VLM-failure fallback (ticket 003 / 004 §4).
# Used only when the VLM stage completely fails; matched against ASR text.
FALLBACK_KEYWORDS = ("买", "卖", "价格", "机制", "上车")


def transcribe(video_path: str, model_path: str, compute_type: str, language: str | None = None) -> list[dict]:
    """Transcribe a video's audio with faster-whisper.

    Returns ``[{start, end, text}, ...]``. Returns ``[]`` on failure (caller falls back).
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        logger.error("[AI-Clip] faster-whisper not installed; ASR stage unavailable.")
        return []

    try:
        logger.info(f"[AI-Clip] Loading ASR model: {model_path} (compute_type={compute_type})")
        # device="cuda" for local GPU; falls back gracefully if CUDA absent.
        model = WhisperModel(model_path, device="cuda", compute_type=compute_type)
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
        logger.success(f"[AI-Clip] ASR done: {len(results)} segments")
        # Release VRAM before the VLM stage loads (ticket 002 §coexistence).
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return results
    except Exception as e:
        logger.error(f"[AI-Clip] ASR failed: {e}")
        return []


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
