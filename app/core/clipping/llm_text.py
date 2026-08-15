"""LLM text-understanding stage: whole-transcript semantic clip detection.

Reads the COMPLETE ASR transcript (not per-segment keyword matching) through a
local text LLM (Qwen2.5-7B-Instruct, 4-bit) so the model sees full context and
can identify live-commerce moments by meaning rather than literal keywords.

Replaces ``keyword_fallback`` as the intelligent text path. VRAM-serial with ASR
and VLM: this module loads its own model and unloads it (``empty_cache``) before
returning, so it never coexists with the VLM on an 8GB card.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from ...utils.logger import logger

# Cloud provider id for the OpenAI-chat-compatible LLM path (ai_clip_llm_provider).
PROVIDER_OPENAI_CHAT = "openai_chat"


def _is_cloud_provider(provider: Any) -> bool:
    """True when the LLM provider config selects the cloud chat-completions path."""
    return str(provider or "").strip().lower() == PROVIDER_OPENAI_CHAT

# Whole-transcript commerce-grounding prompt. Emphasises semantic (not literal)
# detection and timestamp reuse from the provided transcript.
# These are exposed (no leading underscore) so the settings UI can pre-fill them.
SYSTEM_PROMPT = (
    "你是一个直播带货切片助手。你会收到一段完整的直播字幕（带时间戳），"
    "请通读全文，基于上下文语义找出所有\"正式推销商品\"的时段"
    "（主播介绍商品、报价、讲卖点、催下单、对比优惠等带货行为），"
    "忽略纯闲聊、打招呼、预告、和无关内容。"
    "关键词「买/卖/价格/机制/上车/下单/优惠/立减」只是语义示例，"
    "请按实际语义泛化识别同义表达，不要做精确字面匹配。"
    "同一商品的不同卖点（如介绍、报价、催单）可各成一段。"
)

USER_PROMPT_TMPL = (
    "以下是直播字幕，每行格式为 [开始时间 -> 结束时间] 文本：\n\n"
    "{transcript}\n\n"
    "请基于整段语义，输出 JSON 数组，每个元素代表一个带货时段：\n"
    "{\"product\":商品名(无明确名称时写\"商品\"), "
    "\"selling_point\":一句话卖点, "
    "\"start\":\"开始时间\", \"end\":\"结束时间\"}\n"
    "时间必须从上面字幕里已出现过的时间戳中选取（可直接引用某行的开始/结束时间），"
    "不得编造时间。start 必须小于 end。若整段没有带货内容，输出空数组 []。"
    "\n只输出 JSON 数组，不要任何解释文字。"
)

# Backward-compat aliases used inside this module.
_SYSTEM_PROMPT = SYSTEM_PROMPT
_USER_PROMPT_TMPL = USER_PROMPT_TMPL

# Matches HH:MM:SS(.mmm), MM:SS(.xx), or bare seconds. Used to parse LLM output
# timestamps back into float seconds.
_TS_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d{1,3})?)|"  # HH:MM:SS or MM:SS
    r"^(\d+(?:\.\d+)?)$"                                   # bare seconds
)


# --- Model cache: reuse a loaded LLM+tokenizer across segments of the same run.
# Key: (model_path, use_4bit). Avoids re-loading the ~4.5GB Qwen2.5 on every
# segment in segmented recording.
_LLM_CACHE: dict[tuple[str, bool], tuple[Any, Any]] = {}
_LLM_LOCK = threading.Lock()


def _get_llm(model_path: str, use_4bit: bool):
    """Return a cached (model, tokenizer), loading+cacheing on first use."""
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    key = (model_path, use_4bit)
    with _LLM_LOCK:
        cached = _LLM_CACHE.get(key)
        if cached is not None:
            return cached
        logger.info(f"[AI-Clip] Loading text LLM (cached): {model_path} (4bit={use_4bit})")
        load_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
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
        load_kwargs["trust_remote_code"] = True  # needed for GLM-4-9B-Chat etc.
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval()
        _LLM_CACHE[key] = (model, tokenizer)
        return model, tokenizer


def release_llm_models() -> None:
    """Unload all cached LLM models and free VRAM. Call after the run ends."""
    with _LLM_LOCK:
        if not _LLM_CACHE:
            return
        n = len(_LLM_CACHE)
        _LLM_CACHE.clear()
    logger.info(f"[AI-Clip] released {n} cached LLM model(s)")
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def analyze_transcript(
    segments: list[dict],
    model_path: str,
    *,
    use_4bit: bool = True,
    max_new_tokens: int = 2048,
    max_transcript_chars: int = 60000,
    system_prompt: str | None = None,
    user_prompt_tmpl: str | None = None,
) -> list[dict]:
    """Run a cached text LLM over the whole transcript; return global-time clips.

    ``segments`` is the ASR output ``[{start, end, text}, ...]`` in float seconds.
    Returns ``[{product, selling_point, start, end}, ...]`` with ``start``/``end``
    as float seconds. Empty list on failure or when no transcript is available.
    The model is cached across calls; call ``release_llm_models`` when done.

    ``system_prompt`` / ``user_prompt_tmpl`` override the built-in prompts when
    non-empty (user_prompt_tmpl must contain ``{transcript}`` as the placeholder
    where the transcript is injected). Empty/None falls back to the built-ins.
    """
    if not segments:
        return []

    messages = _build_messages(segments, system_prompt, user_prompt_tmpl, max_transcript_chars)

    try:
        import torch  # type: ignore
        model, tokenizer = _get_llm(model_path, use_4bit)
    except ImportError:
        logger.error("[AI-Clip] torch/transformers not installed; LLM text stage unavailable.")
        return []
    except Exception as e:
        logger.error(f"[AI-Clip] LLM load failed: {e}")
        return []

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt")
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        raw = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        logger.debug(f"[AI-Clip] LLM raw output:\n{raw}")
        del inputs, output
        torch.cuda.empty_cache()
        clips = _parse_clips(raw)
        if not clips:
            logger.warning(f"[AI-Clip] LLM output parsed to 0 clips. Raw output was:\n{raw[:1500]}")
        logger.success(f"[AI-Clip] LLM text understanding done: {len(clips)} clip(s)")
        return clips
    except Exception as e:
        logger.error(f"[AI-Clip] LLM inference failed: {e}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return []


def _format_transcript(segments: list[dict]) -> str:
    """Render ASR segments as a timestamped transcript block for the LLM."""
    lines: list[str] = []
    for seg in segments:
        s = _fmt_hms(float(seg.get("start", 0)))
        e = _fmt_hms(float(seg.get("end", 0)))
        text = str(seg.get("text", "")).strip()
        if text:
            lines.append(f"[{s} -> {e}] {text}")
    return "\n".join(lines)


def _fmt_hms(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS`` (no millis, matches the prompt's time format)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_ts(ts: str) -> float:
    """Parse a timestamp string (HH:MM:SS / MM:SS / bare seconds) to float seconds."""
    ts = str(ts).strip()
    # Try HH:MM:SS or MM:SS
    m = re.match(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d{1,3})?)$", ts)
    if m:
        h = int(m.group(1) or 0)
        mn = int(m.group(2))
        s = float(m.group(3))
        return h * 3600 + mn * 60 + s
    # Try bare seconds
    try:
        return float(ts)
    except ValueError:
        return 0.0


def _parse_clips(raw: str) -> list[dict]:
    """Defensively parse free-text JSON the model emits into float-second clips.

    Tries standard JSON first; on failure, falls back to a regex-based extractor
    that tolerates the model's common malformations (missing quotes, extra prose,
    broken delimiters between objects — e.g. ``"end":"00:02:58}`` without the
    closing quote).
    """
    raw = raw.strip()
    for candidate in (raw, _extract_json_array(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            out = _clips_from_list(parsed)
            if out:
                return out
    # Fallback: regex extraction tolerates malformed JSON.
    out = _regex_extract_clips(raw)
    if out:
        logger.info(f"[AI-Clip] JSON parse failed; regex fallback recovered {len(out)} clip(s).")
    return out


def _clips_from_list(parsed: list) -> list[dict]:
    """Convert a parsed JSON list of dicts into float-second clips."""
    out: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        prod = str(item.get("product", "")).strip() or "商品"
        sp = str(item.get("selling_point", "")).strip()
        s = _parse_ts(item.get("start", "0"))
        e = _parse_ts(item.get("end", "0"))
        if e > s:
            out.append({
                "product": prod,
                "selling_point": sp,
                "start": round(s, 2),
                "end": round(e, 2),
            })
    return out


# Regex to extract field values from malformed JSON. Does NOT require a closing
# quote on the value (the model often omits it, e.g. "end":"00:02:58}). Captures
# everything up to the next quote/comma/brace.
_FIELD_RE = re.compile(
    r'"(product|selling_point|start|end)"\s*:\s*"?([^",}]*)',
)


def _regex_extract_clips(text: str) -> list[dict]:
    """Fallback: extract clips from malformed JSON via per-field regex.

    Handles the model's common mistakes: missing closing quotes on values,
    objects separated by ``}`` without proper delimiters, extra prose around
    the array. Each ``{...}`` region is scanned independently.
    """
    out: list[dict] = []
    # Split on '}' to get individual object fragments.
    for frag in text.split("}"):
        frag = frag.strip().lstrip(",").strip()
        if "{" not in frag:
            continue
        fields = dict(_FIELD_RE.findall(frag))
        if not fields:
            continue
        prod = (fields.get("product") or "").strip() or "商品"
        sp = (fields.get("selling_point") or "").strip()
        s = _parse_ts(fields.get("start", "0"))
        e = _parse_ts(fields.get("end", "0"))
        if e > s:
            out.append({
                "product": prod,
                "selling_point": sp,
                "start": round(s, 2),
                "end": round(e, 2),
            })
    return out


def _extract_json_array(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return ""


# --- Cloud path: OpenAI-chat-compatible chat/completions API ----------------
#
# Mirrors analyze_transcript() but sends the same system/user prompts to a
# remote endpoint (SiliconFlow, DeepSeek, OpenAI, local vLLM/ollama, ...) via
# plain httpx — no openai SDK dependency. Reuses _format_transcript and the
# defensive _parse_clips so local and cloud paths stay behaviour-identical.

_API_TIMEOUT_SECONDS = 120.0
_RETRY_BACKOFF_SECONDS = 1.0


def _chat_url(api_base: str) -> str:
    """``{api_base}`` -> ``{api_base}/chat/completions`` (single slash, no dupes)."""
    return str(api_base or "").strip().rstrip("/") + "/chat/completions"


def _build_messages(
    segments: list[dict],
    system_prompt: str | None,
    user_prompt_tmpl: str | None,
    max_transcript_chars: int,
) -> list[dict]:
    """Shared prompt construction for local & cloud paths -> chat messages."""
    sys_p = (system_prompt or "").strip() or _SYSTEM_PROMPT
    usr_tmpl = (user_prompt_tmpl or "").strip() or _USER_PROMPT_TMPL
    if "{transcript}" not in usr_tmpl:
        logger.warning("[AI-Clip] custom user_prompt missing {transcript} placeholder; using built-in.")
        usr_tmpl = _USER_PROMPT_TMPL
    transcript = _format_transcript(segments)
    if len(transcript) > max_transcript_chars:
        logger.warning(
            f"[AI-Clip] transcript very long ({len(transcript)} chars); truncating to {max_transcript_chars}."
        )
        transcript = transcript[:max_transcript_chars]
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": usr_tmpl.replace("{transcript}", transcript)},
    ]


def analyze_transcript_cloud(
    segments: list[dict],
    api_base: str,
    api_key: str,
    api_model: str,
    *,
    max_new_tokens: int = 2048,
    max_transcript_chars: int = 60000,
    system_prompt: str | None = None,
    user_prompt_tmpl: str | None = None,
    max_retries: int = 3,
) -> list[dict]:
    """Cloud twin of ``analyze_transcript``: POST the transcript to an
    OpenAI-chat-compatible endpoint and parse the reply into float-second clips.

    ``api_base`` is everything before ``/chat/completions`` (e.g.
    ``https://api.siliconflow.cn/v1``). Returns ``[]`` on failure; never raises.
    """
    if not segments:
        return []
    api_base = str(api_base or "").strip()
    api_model = str(api_model or "").strip()
    api_key = str(api_key or "").strip()
    if not (api_base and api_model):
        logger.error("[AI-Clip] cloud LLM disabled: api_base/api_model not configured.")
        return []
    if not api_key:
        logger.error("[AI-Clip] cloud LLM disabled: api_key is empty.")
        return []

    import httpx  # main dependency (requirements.txt), no extra install

    messages = _build_messages(segments, system_prompt, user_prompt_tmpl, max_transcript_chars)
    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_new_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    url = _chat_url(api_base)

    # Reasoning models (deepseek-r1/reasoner, glm-z1, qwen-thinking, ...) burn
    # max_tokens on hidden chain-of-thought before writing the visible answer;
    # with a small cap the content comes back empty/truncated with
    # finish_reason=length. On that signal, double the cap and resend.
    _MAX_TOKEN_DOUBLINGS = 3

    last_err: Exception | None = None
    total_attempts = max(1, max_retries)
    doublings = 0
    for attempt in range(1, total_attempts + 1):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=_API_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message", {}) or {}
            finish_reason = str(choice.get("finish_reason") or "")
            raw = str(message.get("content", "") or "")
            reasoning = str(message.get("reasoning_content", "") or "")
            if finish_reason == "length" and attempt < total_attempts and doublings < _MAX_TOKEN_DOUBLINGS:
                new_cap = min(payload["max_tokens"] * 2, 32768)
                if new_cap > payload["max_tokens"]:
                    logger.warning(
                        f"[AI-Clip] cloud LLM hit max_tokens={payload['max_tokens']} "
                        f"(finish_reason=length"
                        + (f", reasoning burned {len(reasoning)} chars" if reasoning else "")
                        + f"); retrying with max_tokens={new_cap}"
                    )
                    payload["max_tokens"] = new_cap
                    doublings += 1
                    continue
            logger.debug(f"[AI-Clip] cloud LLM raw output:\n{raw}")
            clips = _parse_clips(raw)
            if not clips:
                logger.warning(f"[AI-Clip] cloud LLM output parsed to 0 clips. Raw output was:\n{raw[:1500]}")
            logger.success(f"[AI-Clip] cloud LLM text understanding done: {len(clips)} clip(s)")
            return clips
        except Exception as e:
            last_err = e
            # Never log the Authorization header / key.
            logger.warning(f"[AI-Clip] cloud LLM request attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    logger.error(f"[AI-Clip] cloud LLM failed after {max_retries} attempt(s): {last_err}")
    return []
