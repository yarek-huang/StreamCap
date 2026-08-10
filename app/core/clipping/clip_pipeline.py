"""ClipPipeline: orchestrates ASR -> VLM -> FFmpeg cut -> per-clip notify.

Encodes the wayfinder MAP-001 decisions:
  * ticket 004: architecture B (single VLM output, <5s filter, ASR-keyword fallback on VLM failure)
  * ticket 006: clip source = post-transcode mp4; -c copy; padding front-10/back-10; order filter->pad->clamp
  * ticket 007: new module under app/core/clipping/; clips/ subdir + metadata json; retry N=3; self-remux fallback
  * ticket 008: per-clip MessagePusher (pure text); silent on no clips; notify on failures
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from ...messages import message_pusher
from ...models.recording.recording_model import Recording
from ...models.recording.recording_status_model import RecordingStatus
from ...utils.logger import logger
from ...utils.utils import clean_name
from ..runtime.backend_services import BackendServices
from . import asr as asr_stage
from . import vlm as vlm_stage


class ClipPipeline:
    """Stateful holder for one recording's AI-clip run.

    Construct one per finished recording and call :meth:`run`.
    """

    def __init__(self, services: BackendServices, recording: Recording, source_video: str):
        self.services = services
        self.recording = recording
        self.source_video = source_video  # the mp4 to clip on (post-transcode or self-remuxed)

    # -- public -------------------------------------------------------------

    async def run(self) -> None:
        """Execute the full pipeline. Never raises: logs and notifies on failure."""
        cfg = self.services.settings_config.user_config
        if not cfg.get("ai_clip_enabled"):
            return
        try:
            await self._run_inner(cfg)
        except Exception as e:
            logger.error(f"[AI-Clip] pipeline crashed: {e}")
            self._notify_failure("流水线异常", str(e))

    # -- core ---------------------------------------------------------------

    async def _run_inner(self, cfg: dict) -> None:
        self._set_status(RecordingStatus.AI_CLIPPING)
        model_path = cfg.get("ai_clip_mage_vl_model_path", "microsoft/Mage-VL")
        asr_path = cfg.get("ai_clip_asr_model_path", "Systran/faster-whisper-large-v3")
        compute_type = cfg.get("ai_clip_asr_compute_type", "int8_float16")
        window_s = int(cfg.get("ai_clip_window_seconds", 360) or 360)
        overlap_s = int(cfg.get("ai_clip_window_overlap_seconds", 20) or 20)
        use_4bit = bool(cfg.get("ai_clip_use_4bit", True))
        num_frames = int(cfg.get("ai_clip_vlm_num_frames", 16) or 16)
        max_pixels = int(cfg.get("ai_clip_vlm_max_pixels", 150000) or 150000)
        max_new_tokens = int(cfg.get("ai_clip_vlm_max_new_tokens", 512) or 512)
        min_clip = float(cfg.get("ai_clip_min_clip_seconds", 5) or 5)
        pad_start = float(cfg.get("ai_clip_padding_start_seconds", 10) or 10)
        pad_end = float(cfg.get("ai_clip_padding_end_seconds", 10) or 10)
        max_retries = int(cfg.get("ai_clip_max_retries", 3) or 3)

        # 1. ASR stage (ticket 004: ASR runs first)
        segments = await asyncio.to_thread(
            asr_stage.transcribe, self.source_video, asr_path, compute_type
        )
        subtitle = asr_stage.segments_to_srt(segments) if segments else ""

        # 2. VLM stage (ticket 004 architecture B)
        clips = await asyncio.to_thread(
            vlm_stage.analyze_video,
            self.source_video,
            model_path,
            subtitle,
            window_seconds=window_s,
            overlap_seconds=overlap_s,
            use_4bit=use_4bit,
            num_frames=num_frames,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            progress_cb=self._on_window_progress,
        )

        # 2b. Fallback (ticket 004 §4) only on total VLM failure
        if not clips and segments:
            logger.warning("[AI-Clip] VLM produced no clips; using ASR keyword fallback.")
            clips = asr_stage.keyword_fallback(segments)

        if not clips:
            logger.info("[AI-Clip] no commerce clips found; staying silent (ticket 008 §3).")
            self._restore_status()
            return

        # 3. Filter -> pad -> clamp (ticket 006 §3)
        duration = vlm_stage.get_video_duration(self.source_video)
        clips = _apply_filter_pad_clamp(clips, min_clip, pad_start, pad_end, duration)

        # 4. Cut + notify per clip (ticket 007 §6, ticket 008 §1)
        clips_dir = os.path.join(os.path.dirname(self.source_video), "clips")
        os.makedirs(clips_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.source_video))[0]
        success, failed = 0, 0
        for idx, clip in enumerate(clips, 1):
            safe_prod = clean_name(clip.get("product", "clip")) or f"clip{idx}"
            out_name = f"{base}_clip{idx}_{safe_prod}.mp4"
            out_path = os.path.join(clips_dir, out_name)
            ok = await self._cut_with_retry(clip, out_path, max_retries)
            if ok:
                _write_metadata(out_path, clip)
                self._notify_clip(clip, os.path.relpath(out_path, os.path.dirname(self.source_video)))
                success += 1
            else:
                failed += 1
                self._notify_failure(
                    f"切片失败 #{idx} {clip.get('product','')}",
                    f"{self.recording.streamer_name} 片段 {idx} 重试{max_retries}次后失败",
                )
        logger.success(f"[AI-Clip] done: success={success} failed={failed}")
        self._restore_status()

    # -- helpers ------------------------------------------------------------

    async def _cut_with_retry(self, clip: dict, out_path: str, retries: int) -> bool:
        start, end = clip["start"], clip["end"]
        for attempt in range(1, retries + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", self.source_video,
                    "-t", f"{(end - start):.3f}", "-c", "copy",
                    "-avoid_negative_ts", "make_zero", out_path,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                self.services.process_manager.add_process(proc)
                _, stderr = await proc.communicate()
                if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return True
                logger.warning(f"[AI-Clip] cut attempt {attempt} rc={proc.returncode}")
            except Exception as e:
                logger.warning(f"[AI-Clip] cut attempt {attempt} error: {e}")
            await asyncio.sleep(1)
        return False

    def _on_window_progress(self, done: int, total: int, window_start: float) -> None:
        logger.info(f"[AI-Clip] window {done}/{total} (start={window_start:.1f}s)")
        self.recording.status_info = RecordingStatus.AI_CLIPPING
        try:
            self.services.broadcast_card_update(self.recording)
        except Exception:
            pass

    def _set_status(self, status: str) -> None:
        self.recording.status_info = status
        try:
            self.services.broadcast_card_update(self.recording)
        except Exception:
            pass

    def _restore_status(self) -> None:
        # return to a sensible resting status
        if self.recording.monitor_status:
            self.recording.status_info = RecordingStatus.MONITORING
        else:
            self.recording.status_info = RecordingStatus.STOPPED_MONITORING
        try:
            self.services.broadcast_card_update(self.recording)
        except Exception:
            pass

    # -- notifications (ticket 008) -----------------------------------------

    def _should_notify(self) -> bool:
        cfg = self.services.settings_config.user_config
        if not cfg.get("ai_clip_notification_enabled"):
            return False
        return message_pusher.MessagePusher._get_push_channels() and any(
            cfg.get(ch) for ch in message_pusher.MessagePusher._get_push_channels()
        )

    def _push(self, title: str, content: str) -> None:
        if not self._should_notify():
            return
        mp = message_pusher.MessagePusher(self.services.settings_config)
        self.services.run_coro(mp.push_messages(title, content))

    def _notify_clip(self, clip: dict, rel_path: str) -> None:
        cfg = self.services.settings_config.user_config
        template = cfg.get("ai_clip_custom_content") or \
            "[streamer] | [title] | 切片：[product] - [selling_point] ([time]) [clip_path]"
        content = (
            template
            .replace("[streamer]", self.recording.streamer_name or "")
            .replace("[title]", self.recording.live_title or self.recording.title or "")
            .replace("[product]", clip.get("product", ""))
            .replace("[selling_point]", clip.get("selling_point", ""))
            .replace("[time]", f"{_fmt_sec(clip['start'])}-{_fmt_sec(clip['end'])}")
            .replace("[clip_path]", rel_path)
        )
        title = (cfg.get("custom_notification_title") or "").strip() or "AI切片完成"
        self._push(title, content)

    def _notify_failure(self, what: str, detail: str) -> None:
        content = f"[{self.recording.streamer_name}] AI切片{what}：{detail}"
        self._push("AI切片失败", content)


# -- module-level helpers ----------------------------------------------------

def _apply_filter_pad_clamp(
    clips: list[dict], min_clip: float, pad_start: float, pad_end: float, duration: float
) -> list[dict]:
    """Ticket 006 §3: filter <min_clip (on raw) -> pad -> clamp to [0, duration]."""
    out: list[dict] = []
    for c in clips:
        s, e = float(c["start"]), float(c["end"])
        if e - s < min_clip:
            continue  # filter first, before padding
        s = max(0.0, s - pad_start)
        e = duration if duration <= 0 else min(duration, e + pad_end)
        out.append({**c, "start": round(s, 2), "end": round(e, 2)})
    return out


def _write_metadata(clip_path: str, clip: dict) -> None:
    meta_path = os.path.splitext(clip_path)[0] + ".json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(clip, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[AI-Clip] metadata write failed: {e}")


def _fmt_sec(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def self_remux_to_mp4(src_path: str) -> str | None:
    """Ticket 007 §5: remux non-ts source to a temp mp4 so the pipeline has an mp4 to clip on.

    Returns the mp4 path or None on failure.
    """
    if src_path.lower().endswith(".mp4"):
        return src_path
    out_path = os.path.splitext(src_path)[0] + ".aiclip.mp4"
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-c", "copy", "-f", "mp4", out_path],
            capture_output=True, timeout=600,
        )
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception as e:
        logger.error(f"[AI-Clip] self-remux failed: {e}")
    return None
