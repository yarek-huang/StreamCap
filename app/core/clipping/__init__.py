"""AI-driven live-commerce clip pipeline.

Resolves wayfinder map MAP-001. Pipeline: recording finishes -> ASR (faster-whisper)
-> VLM (Mage-VL, fed ASR transcript as subtitles, joint judgment) -> FFmpeg clips
(``-c copy``) -> per-clip notification via MessagePusher.

Decisions encoded here: see ``.wayfinder/MAP-001-ai-clip-pipeline.md`` and tickets 001-008.
"""
from .clip_pipeline import ClipPipeline

__all__ = ["ClipPipeline"]
