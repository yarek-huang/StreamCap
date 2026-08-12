"""VLM stage (tickets 001, 003, 004, 005): Mage-VL local offline inference.

Architecture B (ticket 004): ASR transcript is fed to the VLM as subtitle input so the
model performs joint visual+speech judgment and emits a single clip list (no post-fusion).

Sliding windows (ticket 005): 5-8 min each, 15-30s overlap; each window's emitted
``MM:SS.xx`` timestamps are offset by the window start to recover global video time.
Frame-sampling backend (cv2 uniform sampling) — avoids the external ``cv-preinfer``
binary required by the HEVC codec backend, and the DCVC compile required by the neural
backend, so the VLM runs on Windows with zero external binaries.
Timestamps come from prompt-generated free text (ticket 001), parsed defensively.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ...utils.logger import logger


def get_video_duration(path: str) -> float:
    """Return duration in seconds via ``ffprobe`` (returns 0 on failure)."""
    import json as _json
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        data = _json.loads(out.stdout or "{}")
        return float(data.get("format", {}).get("duration", 0) or 0)
    except Exception as e:
        logger.error(f"[AI-Clip] ffprobe duration failed: {e}")
        return 0.0


# Prompt embodying the live-commerce taxonomy (ticket 003 §Resolution / Prompt 草案).
_GROUNDING_PROMPT = (
    "你是一个直播带货切片助手。观看这段直播录像，找出所有\"正式推销商品\"的时段"
    "（主播拿起/展示商品，或在介绍、报价、催下单等带货行为），忽略纯闲聊和预热。\n"
    "关键词「买/卖/价格/机制/上车」仅作为带货语义的示例，请按语义泛化识别同义表达，"
    "不要做精确字面匹配。同时参考下方字幕文本判断语音内容。\n"
    "对每个带货时段按\"重点卖点\"拆分：同一商品的不同卖点各成一段。\n"
    "输出 JSON 数组，每个元素：{\"product\":商品名, \"selling_point\":一句话卖点, "
    "\"start\":\"MM:SS.xx\", \"end\":\"MM:SS.xx\"}。时间只用本窗采样到的帧对应的时间戳，不得编造。"
    "若本窗没有带货内容，输出空数组 []。只输出 JSON，不要其他文字。"
)

_TS_RE = re.compile(r"(\d{1,2}):(\d{2}(?:\.\d{1,3})?)")


def analyze_video(
    video_path: str,
    model_path: str,
    subtitle_text: str,
    *,
    window_seconds: int = 360,
    overlap_seconds: int = 20,
    use_4bit: bool = True,
    num_frames: int = 16,
    max_pixels: int = 150000,
    max_new_tokens: int = 512,
    progress_cb=None,
) -> list[dict]:
    """Run Mage-VL over the whole video in sliding windows, return global-time clips.

    ``progress_cb(done, total, window_start)`` is called per window for UI feedback
    (ticket 005 §5). Returns ``[{product, selling_point, start, end}, ...]`` in seconds
    as floats. Empty list on total failure (caller triggers ASR fallback).
    """
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
    except ImportError:
        logger.error("[AI-Clip] torch/transformers not installed; VLM stage unavailable.")
        return []

    duration = get_video_duration(video_path)
    if duration <= 0:
        logger.error(f"[AI-Clip] cannot determine video duration: {video_path}")
        return []

    try:
        logger.info(f"[AI-Clip] Loading VLM: {model_path} (4bit={use_4bit})")
        load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": "auto"}
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except Exception as e:
                logger.warning(f"[AI-Clip] 4-bit unavailable ({e}); loading at full precision.")
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto"
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Configure pixel budget on EVERY processor sub-object that carries a
        # max_pixels attribute. The image_processor (Qwen2VLImageProcessor) does
        # the final smart_resize on pixel values, the video_processor does the
        # frame-level resize; both default to ~4,000,000 px/frame (from
        # preprocessor_config.json) which blows up 8GB VRAM. Setting min<max
        # avoids smart_resize logic conflict.
        _budget_min = min(max_pixels // 4, 200704)
        for _attr in ("video_processor", "image_processor"):
            _obj = getattr(processor, _attr, None)
            if _obj is not None:
                try:
                    _obj.max_pixels = max_pixels
                    _obj.min_pixels = _budget_min
                    logger.info(f"[AI-Clip] {_attr} pixel budget: max={max_pixels} min={_budget_min}")
                except Exception as _e:
                    logger.warning(f"[AI-Clip] could not set pixel budget on {_attr}: {_e}")
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval()
    except Exception as e:
        logger.error(f"[AI-Clip] VLM load failed: {e}")
        return []

    clips: list[dict] = []
    window_start = 0.0
    window_idx = 0
    total_windows = max(1, int(duration // max(1, (window_seconds - overlap_seconds))) + 1)

    try:
        while window_start < duration:
            window_end = min(window_start + window_seconds, duration)
            window_idx += 1
            if progress_cb:
                try:
                    progress_cb(window_idx, total_windows, window_start)
                except Exception:
                    pass
            try:
                window_clips = _run_window(
                    model, processor, video_path, window_start, window_end, subtitle_text,
                    num_frames=num_frames, max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                logger.error(f"[AI-Clip] window {window_idx} failed: {e}")
                window_clips = []
            for c in window_clips:
                # offset local MM:SS.xx -> global seconds
                c["start"] = _ts_to_sec(c.get("start", "00:00.00")) + window_start
                c["end"] = _ts_to_sec(c.get("end", "00:00.00")) + window_start
                if c["end"] <= c["start"]:
                    continue
                c["start"] = round(c["start"], 2)
                c["end"] = round(c["end"], 2)
                clips.append(c)
            step = max(1, window_seconds - overlap_seconds)
            window_start += step
    finally:
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return clips


def _run_window(
    model, processor, video_path: str, start: float, end: float, subtitle_text: str,
    *, num_frames: int = 16, max_new_tokens: int = 512,
) -> list[dict]:
    """Inference one window via the frame-sampling backend (default ``frames``).

    The window is physically cut to a temp mp4 first (``ffmpeg -ss -t -c copy``) so the
    VLM's ``video_processor`` receives a self-contained file path and handles frame
    extraction + ``smart_resize`` itself. Passing a pre-sampled PIL list is NOT correct:
    the processor re-processes it and the per-frame pixel budget (``max_pixels``) ends
    up feeding an unbounded tensor shape -> OOM on a ~142 GiB allocation.

    ``num_frames`` (cap) is the frame-count knob for VRAM; ``max_pixels`` is the
    per-frame pixel budget (applied directly on ``video_processor`` because the
    processor's ``__call__`` does not propagate it to the video path). On 8GB
    cards keep ``num_frames`` low (8-16) and ``max_pixels`` modest.
    """
    import os
    import tempfile

    import torch  # type: ignore

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.empty_cache()

    fd, window_path = tempfile.mkstemp(prefix="aiclip_window_", suffix=".mp4")
    os.close(fd)
    try:
        _cut_window_file(video_path, start, end, window_path)
        user_text = f"{_GROUNDING_PROMPT}\n\n字幕参考：\n{subtitle_text}" if subtitle_text else _GROUNDING_PROMPT
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": user_text},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Frame-sampling path: pass the window mp4 path as `videos=[path]`. The
        # processor's video_processor extracts `num_frames` frames and applies
        # smart_resize bounded by max_pixels/min_pixels set directly on the vp
        # in analyze_video (the __call__ kwargs do NOT propagate pixel budget).
        inputs = processor(
            text=[text],
            videos=[window_path],
            num_frames=num_frames,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        del inputs, output
        torch.cuda.empty_cache()
        return _parse_clips(raw)
    finally:
        try:
            os.remove(window_path)
        except OSError:
            pass


def _cut_window_file(src: str, start: float, end: float, dst: str) -> None:
    """Cut ``[start,end]`` of ``src`` to ``dst`` via ``ffmpeg -ss -t -c copy``."""
    import subprocess
    dur = max(0.1, end - start)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
         "-c", "copy", "-avoid_negative_ts", "make_zero", dst],
        capture_output=True, timeout=300,
    )


def _parse_clips(raw: str) -> list[dict]:
    """Defensively parse free-text JSON the model emits (ticket 001 §4 risk)."""
    raw = raw.strip()
    # try direct JSON, then first JSON array substring
    for candidate in (raw, _extract_json_array(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                prod = str(item.get("product", "")).strip()
                sp = str(item.get("selling_point", "")).strip()
                s = item.get("start")
                e = item.get("end")
                if prod and s is not None and e is not None:
                    out.append({"product": prod, "selling_point": sp, "start": str(s), "end": str(e)})
            if out:
                return out
    return []


def _extract_json_array(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return ""


def _ts_to_sec(ts: str) -> float:
    m = _TS_RE.match(str(ts).strip())
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))
