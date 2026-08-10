# AI 切片功能 — 部署文档

StreamCap 直播带货 AI 切片流水线：录完 → faster-whisper(ASR) → Mage-VL(VLM，喂入字幕联合判断) → FFmpeg `-c copy` 切片 → 逐片通知。

> 设计依据见 `.wayfinder/MAP-001-ai-clip-pipeline.md` 及 8 张决策票。本文只讲部署。

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10/11（本功能仅面向 Windows 桌面端） |
| 显卡 | **NVIDIA 独显，显存 ≥8GB**（RTX 3060/4060 及以上） |
| Python | 3.10+（StreamCap 本身要求） |
| FFmpeg/FFprobe | 已安装并在 PATH（StreamCap 录制已依赖） |
| 磁盘 | 模型权重 ~10GB（Mage-VL 4B + faster-whisper large-v3）+ 切片产物空间 |

> ⚠️ 显存 8GB 是**下限**：Mage-VL BF16 权重本身 ~9.5GB，**必须开 4-bit 量化**才能跑（设置项「4-bit 量化」默认开）。12GB 更稳。

## 2. 安装 Python 依赖

AI 切片依赖较重且版本敏感（`transformers>=5.7` 较新），**不放在主 `requirements.txt` 里**，单独装：

```bash
pip install -r requirements-ai-clip.txt
```

等价于逐项安装：

```bash
# 视觉模型（transformers>=5.7 是 Mage-VL 的硬依赖，较新）
pip install "transformers>=5.7" accelerate

# 4-bit 量化（8GB 显存必装；Windows 原生若装不上，见下方 WSL2 方案）
pip install bitsandbytes

# 语音模型（CTranslate2 后端，不依赖 torch，与 VLM 共存无版本冲突）
pip install faster-whisper

# 视频解码（Mage-VL 帧采样 / HEVC 后端需要）
pip install opencv-python numpy Pillow decord
```

> `faster-whisper` 跑在 CTranslate2 上，依赖仅 setuptools/numpy/pyyaml，**不会与 Mage-VL 的 torch/CUDA 版本冲突**——这是它能和 VLM 同环境共存的关键。

## 3. 模型权重

### 3.1 在线自动下载（首次运行）
首次启用时，`transformers` 与 `faster-whisper` 会自动从 HuggingFace 下载：
- Mage-VL：`microsoft/Mage-VL`（约 8GB，含 neural_codec 子包）
- ASR：`Systran/faster-whisper-large-v3`（约 3GB）

下载缓存在默认 HF 缓存目录（Windows: `C:\Users\<你>\.cache\huggingface\hub`）。

### 3.2 离线/手动下载（无外网或下载慢）
把模型下到本地任意目录，然后在 StreamCap 设置里把「模型路径」改成**本地绝对路径**：

```
Mage-VL 模型路径:  D:\models\Mage-VL
ASR 模型路径:      D:\models\faster-whisper-large-v3
```

可用 `huggingface-cli` 下载：
```bash
pip install huggingface_hub
huggingface-cli download microsoft/Mage-VL --local-dir D:\models\Mage-VL
huggingface-cli download Systran/faster-whisper-large-v3 --local-dir D:\models\faster-whisper-large-v3
```

## 4. bitsandbytes 在 Windows 原生装不上的处理（4-bit 量化）

若 `pip install bitsandbytes` 在原生 Windows 报错（无匹配 wheel），有两个出路：

- **方案 A（推荐，最稳）**：把 StreamCap 跑在 **WSL2** 里。Mage-VL 的 neural DCVC-RT 与 bitsandbytes 在 WSL2 下都顺。按 StreamCap 的 Web 模式运行（`.env` 设 `PLATFORM=web`，`python main.py --web`），从 Windows 浏览器访问。
- **方案 B（留在原生 Windows）**：不装 bitsandbytes，在设置里**关闭「4-bit 量化」**，模型以 BF16 加载——但 **8GB 显存大概率 OOM**，仅 12GB+ 显存可用此方案。或换用社区提供的 Windows bitsandbytes 预编译 wheel。

## 5. 启用与配置

1. 启动 StreamCap：`python main.py`
2. 进入 **设置 → 录制设置**，滚到底部「AI 切片」区块：
   - 打开「启用 AI 切片」（总开关 `ai_clip_enabled`）
   - 按需打开「AI 切片结果通知」并在上方「消息推送」里配好渠道（钉钉/飞书/Bark/Telegram 等）
   - 其余参数见 `docs/ai_clip_config.md`
3. 默认参数已按 8GB 显存 + 带货场景调好（5–8 分钟窗、padding 前10后10、最短片段5秒、重试3次），通常无需改。

## 6. 运行流程（录完自动触发）

```
直播结束 → StreamCap 转码(ts→mp4) → [AI 切片链]
  ASR(faster-whisper, int8_float16, ~3GB) 出带时间戳字幕
    → 释放显存
  VLM(Mage-VL, 4-bit, 5-8分钟HEVC窗, 字幕作输入联合判断) 出 {product,selling_point,start,end}
    → 过滤<5秒 → padding前10后10 → clamp
  FFmpeg -c copy 切片 → <原录像目录>/clips/*.mp4 + 同名元信息.json
  每切出一片 → MessagePusher 逐片通知
  某片失败 → 重试3次 → 仍失败则发失败通知、跳过继续
  无任何带货内容 → 静默不发
```

UI 录制卡片会显示 **「AI_CLIPPING X/Y」** 进度。

## 7. 验证（首次部署后建议）

1. 录一段短的带货直播（5–10 分钟）。
2. 录完后看日志 `[AI-Clip]` 前缀：应依次出现 ASR done → VLM windows → 切片 done。
3. 检查录像目录下 `clips/` 是否生成切片 + `.json` 元信息。
4. 若开了通知，确认渠道收到逐片消息。

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `CUDA out of memory` | 显存不足。确认「4-bit 量化」已开；调小「VLM 窗口大小」（如 300）；或换 12GB+ 显存/WSL2。 |
| `faster-whisper not installed` | 第 2 步依赖没装；`pip install faster-whisper`。 |
| `transformers` 报版本不符 | Mage-VL 要求 `transformers>=5.7`，升级：`pip install -U "transformers>=5.7"`。 |
| 切片边界差几秒 | `-c copy` 只能切到最近关键帧，属正常；已用 padding 前10后10 弥补。要帧级精确需改重编码（见 `vlm.py` 切法，当前为无损 copy）。 |
| 通知没收到 | 确认「AI 切片结果通知」开 + 至少一个推送渠道开 + 渠道配置正确。无切片时**静默不发**是设计如此。 |
| FLV 录制没 mp4 | 流水线会自动 remux 出临时 `.aiclip.mp4` 再切，无需开「转码 mp4」也行。 |
| 时间戳错位 | 多窗口时间已自动累加偏移；若仍错位，检查录像本身是否被分段录制（分段文件需各自跑）。 |

## 9. 已知限制（移交实现的验证项）

- 5–8 分钟窗在 8GB+4-bit 下的实际 VRAM/token 上限**未经官方验证**；若 OOM 会自动按 fallback 降帧/缩窗。
- Mage-VL 的时间戳是 prompt 生成的自由文本，已做防御性 JSON 解析，但仍可能偶发解析失败（该窗跳过）。
- 多场录制同时结束时**串行排队**（同卡无法并行驻留两模型），队列顺序为先到先跑。
