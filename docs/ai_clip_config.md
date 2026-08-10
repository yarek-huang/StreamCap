# AI 切片功能 — 配置文档

所有配置项写在 `config/default_settings.json`，可在 StreamCap **设置 → 录制设置 → AI 切片** 区块图形化修改，改完自动保存。也可直接编辑 JSON。

> 部署见 `docs/ai_clip_deploy.md`。设计依据见 `.wayfinder/MAP-001-ai-clip-pipeline.md`。

## 配置项总览

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ai_clip_enabled` | bool | `false` | **总开关**。开 = 录完后自动跑 AI 切片流水线。 |
| `ai_clip_notification_enabled` | bool | `false` | 切片结果通知开关（独立于直播开播/结束通知）。开 + 至少一个推送渠道开才发。 |
| `ai_clip_mage_vl_model_path` | str | `microsoft/Mage-VL` | VLM 模型。HuggingFace id 或本地绝对路径。 |
| `ai_clip_asr_model_path` | str | `Systran/faster-whisper-large-v3` | ASR 模型。HuggingFace id 或本地绝对路径。 |
| `ai_clip_asr_compute_type` | str | `int8_float16` | ASR 精度。`int8_float16`(省显存,~3GB) / `float16`(~4.5GB) / `int8`。8GB 显存用默认。 |
| `ai_clip_use_4bit` | bool | `true` | VLM 4-bit 量化。**8GB 显存必开**（不开 BF16 权重 ~9.5GB 会 OOM）。12GB+ 可关。 |
| `ai_clip_window_seconds` | str→int | `360` | VLM 滑动窗大小（秒）。**5–8 分钟=300–480**。窗要足够大让模型判断"换品"。 |
| `ai_clip_window_overlap_seconds` | str→int | `20` | 窗重叠（秒）。15–30。避免事件被切在窗边界。 |
| `ai_clip_padding_start_seconds` | str→float | `10` | 切片前 padding（秒）。在模型给的 start 前外扩。 |
| `ai_clip_padding_end_seconds` | str→float | `10` | 切片后 padding（秒）。在模型给的 end 后外扩。 |
| `ai_clip_min_clip_seconds` | str→float | `5` | 最短片段（秒）。**模型原始输出** < 此值则丢弃（碎片/抖动过滤）。在 padding 之前施加。 |
| `ai_clip_max_retries` | str→int | `3` | 单片 ffmpeg 切失败的重试次数，超过则跳过并发失败通知。 |
| `ai_clip_custom_content` | str | 见下 | 切片通知文案模板，支持占位符。 |

## 通知文案模板占位符

`ai_clip_custom_content` 默认值：
```
[streamer] | [title] | 切片：[product] - [selling_point] ([time]) [clip_path]
```

| 占位符 | 替换为 |
|---|---|
| `[streamer]` | 主播名 |
| `[title]` | 直播标题（无则录制的 title） |
| `[product]` | 该片段商品名（VLM 判定） |
| `[selling_point]` | 该片段一句话卖点 |
| `[time]` | 片段时间区间，如 `02:30.00-03:15.00` |
| `[clip_path]` | 切片**相对原录像目录**的路径（如 `clips/xxx_clip1_商品.mp4`）。完整绝对路径在同名 `.json` 元信息里。 |

> 通知复用现有渠道（钉钉/微信/飞书/Bark/ntfy/Telegram/邮件/Server酱），**纯文本无配图**。通知文本用相对路径是为了避免长 Windows 绝对路径撑爆渠道文本上限。

## 默认值调参建议

### 显存相关
- **8GB 显存**：保持 `ai_clip_use_4bit=true`、`ai_clip_asr_compute_type=int8_float16`、`ai_clip_window_seconds=360`。若仍 OOM，把窗口降到 `300`。
- **12GB+ 显存**：可试 `ai_clip_use_4bit=false`（精度更高，但更吃显存）。

### 切片质量相关
- 切片**偏短/切过头**：调大 `ai_clip_padding_start/end_seconds`（如 15）。
- 切片**太碎/太多误切**：调大 `ai_clip_min_clip_seconds`（如 8）。
- 切片**边界差几秒**：`-c copy` 只能对齐关键帧，属正常；padding 就是为此设计，别改重编码除非接受慢。

### 窗口相关
- **换品判断错乱**（同品被切成多段、或不同品被并）：`ai_clip_window_seconds` 调大（如 420–480），给模型更长上下文。
- **处理太慢但能接受**：保持现状（设计上就容忍任意时长，后台跑）。

## 触发与降级逻辑（无需配置，自动）

- **触发**：录制结束、转码完成后自动跑（若 `ai_clip_enabled=true`）。不与转码竞争。
- **VLM 完全失败**：自动降级为 ASR 关键词硬匹配（关键词：买/卖/价格/机制/上车），命中即切（`product` 标记为"ASR兜底"）。
- **无任何带货内容**：静默不发通知。
- **单片失败**：重试 N 次后跳过，继续切其他片，并发一条失败通知。

## 产物组织

切片落在**原录像同目录**的 `clips/` 子目录：
```
<录像目录>/
  原录像.mp4
  clips/
    原录像_clip1_商品A.mp4
    原录像_clip1_商品A.json      ← 元信息 {product, selling_point, start, end}
    原录像_clip2_商品B.mp4
    原录像_clip2_商品B.json
```
命名规则：`<原文件名>_clip<序号>_<商品名>.mp4`，商品名做了文件名安全化。

## 通知渠道配置

AI 切片通知**不新增渠道**，复用「设置 → 消息推送」里已配的渠道。只要那里开了至少一个渠道 + 开了 `ai_clip_notification_enabled`，逐片就会推到所有已开渠道。

各渠道的具体配置（webhook URL、token 等）见 StreamCap 原有的消息推送设置，不在本文重复。
