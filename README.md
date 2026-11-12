<div align="center">

# 🚀 Alili MiniMax H3 集成 (Alili-MiniMax-H3)

### ComfyUI 原生 MiniMax H3 视听生成 · 一体化条件节点 · 双时钟采样 · 自动分段 · 潜空间放大 · 大模型提示词优化

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blueviolet?style=flat-square)](https://github.com/comfyanonymous/ComfyUI)
[![Video](https://img.shields.io/badge/Task-Text%2FImage%2FRef--to--Video-FF4D4D?style=flat-square)]()
[![Author](https://img.shields.io/badge/Author-阿里里-brightgreen?style=flat-square)]()
[![License](https://img.shields.io/badge/License-Free%20for%20All-brightgreen?style=flat-square)]()

**把开源 MiniMax H3 音视频模型接进 ComfyUI：文生视频、图生视频、参考生视频、首尾帧、口型驱动，配套双时钟采样、长视频自动分段、潜空间放大，以及调用大模型的提示词优化器。**

[📺 B站视频教程](#-b站视频教程) • [✨ 功能亮点](#-功能亮点) • [📸 图片展示](#-图片展示) • [🔧 节点与文件](#-节点与文件-)

---

</div>

## 📖 项目简介

**Alili MiniMax H3 集成** 是一套面向 ComfyUI 原生 MiniMax H3 音视频模型的完整工作流插件。它提供：

- 一个**一体化条件输入节点**（文 / 图 / 参考 → 视听条件），把扩散模型留在外部，方便你接任意兼容采样器；
- 与之配套的**双时钟采样器**、**自动分段采样器**、**潜空间放大器**；
- 一个调用大模型的**提示词优化后端**（把你的一句话 / 参考图扩写成 MiniMax H3 生产级提示词）。

作者：**阿里里**。免费向所有人开放使用。

---

## ✨ 功能亮点

### 🎬 大模型 / 一体化条件节点 `AliliMiniMaxH3Integration`

- 支持 **文生视频（T2VA）**、**图生视频（I2VA）**、**首尾帧（FL2VA）**、**参考生视频（Ref2VA）**、**混合（Hybrid）** 等任务，自动（`auto`）识别。
- 两种主模式：**文本关键帧（text_keyframes）** 与 **全参考（all_reference）**。
- 素材槽位：首帧 / 尾帧、最多 **9 张参考图**、3 个参考视频、3 个参考音频、1 个 Hybrid 音频。
- 画布控制：**自适应 / 16:9 / 9:16 / 3:2 / 2:3 / 4:3 / 3:4 / 1:1 / 21:9** 比例，百万像素 0.2–2.0，时长 2–30 秒。
- **分镜提示词自动拆分**：带 `[Shot k]` / `【镜头 k】` 标记的整份提示词会被拆成 N 份完整提示词，分别构造条件后注入。
- 音频模式：`native` / `lock_source`（原声锁定）/ `reference_only` / `remix_source`；支持音频降噪强度、驱动音频序号、严格提示词标签等。

### ⏱️ 双时钟采样器 `AliliMiniMaxH3DualClock`

- 基于 T8 双时钟设计，**视频 / 音频独立 sigma 偏移**（`shift_video` / `shift_audio`），兼容当前 ComfyUI 的 `FLOW_AV`。
- 对「锁定的源音频」做干净的 inpaint 处理，避免口型同步被破坏；与下游二采 / 潜空间放大正确联动。
- 内置 GC 清理守卫，规避部分 ComfyUI 版本在动态换模型（H3 → 放大 → H3）时的 `real_model` 空引用问题。

### 🧩 自动分段采样器 `AliliMiniMaxH3AutoSegment`（首采 / 二采）

- 把长视频拆成若干短段依次生成或修复：**每段开头锁定上一段的尾巴**，解码后切掉借来的前缀再拼接。
- 同时输出**普通潜空间**与**纯净降噪潜空间**两条路（解决二采发糊的关键）。
- 支持空间分块（同二采分块采样器）、接缝重绘强度、干净前缀条件、逐段解码输出。

### 🔍 潜空间放大器 `AliliMinimaxH3LatentUpscaler`

- 专为 MiniMax H3 设计的 **3D 潜空间放大模型**（`LatentResizer3D`）。
- 三种放大模式：**按倍率（scale）** / **目标尺寸** / **百万像素**；自动对齐 32 倍数。
- 显存不足时**自动降级分块**（32→4 逐级尝试），支持 CUDA / ROCm / CPU，可指定 fp16 / bf16 / fp32。

### 🤖 大模型提示词优化（Prompt Optimizer）

- 调用 LLM 把一句话 / 参考图扩写成 MiniMax H3 生产级提示词；支持系统提示词自动构造**六大分区 / 对白 `<d>` / 主体 `(S1)` / 分镜 `[Shot N]`** 等严格结构。
- 内置多平台 API 客户端（`h3_api.py`）：**DeepSeek / SiliconFlow / Gemini / 阿里云百炼(通义千问) / OpenAI / OpenRouter / 智谱 GLM / Ollama / RunningHub / 自定义(OpenAI 兼容)**。
- 输出语言可选**中文 / English**；异步任务 + 取消，不阻塞界面。
- 凭据统一存于 `~/.ai_cache/ai_api_config.json`（与 Alili-Image-Workbench 共享同款客户端）。

### 🔗 网页⇄节点联动（适配器）

- `AliliMiniMaxH3IntegrationAdapter` 把一体化节点输出展开为：正向条件、`av_latent`、视频 VAE、音频 VAE、模型数量、mux 音频、是否原声、成品提示词、媒体映射 JSON、分段报告——直接喂给原生采样器与解码节点。

### 🌐 多语言

- 内置 `locales/en` 与 `locales/zh`（含 `main.json` / `nodeDefs.json`），节点名称与说明随 ComfyUI 语言切换。

---

## 📸 图片展示

> 💡 **把你自己准备的截图放进本目录下的 `images/` 文件夹，并替换下面的文件名即可自动生效。**

| 条件节点面板 | 采样 / 分段工作流 |
| :---: | :---: |
| ![条件节点](images/preview-integration.png) | ![采样分段](images/preview-sampling.png) |
| 文 / 图 / 参考生视频配置 | 双时钟采样 + 自动分段 |

| 潜空间放大 | 提示词优化 |
| :---: | :---: |
| ![放大](images/preview-upscale.png) | ![优化](images/preview-optimizer.png) |
| 3D 潜空间放大 | 调用大模型扩写提示词 |

---

## 📺 B站视频教程

> 完整视频教程请关注作者 B 站空间，分集上线后会在此补充对应链接：

👉 **B 站教程空间**：https://space.bilibili.com/67359822

---

## 📦 安装

1. 把整个 `Alili-MiniMax-H3` 文件夹放进 ComfyUI 的 `custom_nodes/` 目录：
   ```
   ComfyUI/custom_nodes/Alili-MiniMax-H3/
   ```
2. 安装依赖（如尚未安装）：`aiohttp`、ComfyUI 自带 `comfy_api` 等。
3. 重启 ComfyUI，在画布右键 → `Alili/MiniMax H3 Integration` 分类下找到各节点。
4. 提示词优化需在设置里配置大模型 API（齿轮按钮，凭据写入 `~/.ai_cache/ai_api_config.json`）。

> 本插件只负责**条件构造与采样**；扩散模型（MiniMax H3）与视频 / 音频 VAE 仍由 ComfyUI 原生加载，保持外部可控。

---

## 🔧 节点与文件

### 核心节点

| 节点名 | 类别 | 说明 |
| :--- | :--- | :--- |
| **Alili MiniMax H3 Integration** | `Alili/MiniMax H3 Integration` | 一体化条件输入：文 / 图 / 参考 → 视听条件（扩散模型留外部） |
| **Alili H3 自动分段采样器 (首采/二采)** | `Alili/MiniMax H3 Integration` | 长视频拆段首尾接力、双路潜空间输出 |
| **MiniMax H3 Dual Clock** | `Alili/MiniMax H3 Integration` | 双时钟采样器（视频 / 音频独立 sigma 偏移） |
| **Minimax H3 Latent Upscaler** | `Alili/MiniMax H3 Integration/Upscaling` | 3D 潜空间放大 |
| **Alili MiniMax H3 Integration Adapter** | `Alili/MiniMax H3 Integration` | 把一体化输出展开为正向条件 / VAE / 音频 / 报告等 |

### 文件结构 🧭

```
Alili-MiniMax-H3/
├── __init__.py              # 入口：comfy_entrypoint() 注册扩展
├── nodes.py                 # 节点定义：Integration / Adapter / DualClock / AutoSegment / Upscaler
├── core.py                  # 帧数对齐（17n+5 网格）、空 AV 潜空间、音频 VAE 编码
├── conditioning.py          # 条件构造（文本 / 图像 / 视频 / 音频参考）
├── sampling.py              # 双时钟采样器（shift_video / shift_audio，干净锁定音频）
├── segment_sampler.py       # 自动分段采样（首采 / 二采，双路潜空间）
├── tiled_sampler.py         # 空间分块采样
├── h3_sequencer.py          # 分段规划 / 接力帧连续性
├── latent_upscaler.py       # 3D 潜空间放大模型（LatentResizer3D）
├── prompt_optimizer.py      # 提示词优化后端（六大分区 / 对白<d> / 主体(S1) / 分镜[Shot N]）
├── h3_api.py                # 大模型 API 客户端（10+ 平台，配置 ~/.ai_cache/ai_api_config.json）
├── utils.py                 # 工具函数（AV latent 解析、掩码拆分等）
├── locales/                 # 多语言
│   ├── en/main.json, nodeDefs.json
│   └── zh/main.json, nodeDefs.json
└── web/js/
    └── minimax_h3_integration.js   # 条件节点前端面板
```

---

## ❓ 常见问题

**Q：节点里没有 MiniMax H3 扩散模型？**
> 本插件**不内置**扩散模型。请在 ComfyUI 原生 `Load Diffusion Model` 里加载 MiniMax H3 权重，再接本插件的条件输出。

**Q：提示词优化提示「未检测到 API Key」？**
> 点节点面板的设置齿轮，选择平台并填入对应 `API Key`（凭据写入 `~/.ai_cache/ai_api_config.json`）。Ollama / 自定义平台可免 Key。

**Q：放大时报显存不足？**
> 插件会**自动降级分块**；也可在节点上手动调小目标尺寸 / 百万像素，或把 `device` 切到 `cpu`（更慢但更省显存）。

**Q：长视频怎么不爆显存？**
> 用 **自动分段采样器** 把总时长拆成 4–20 秒的小段依次生成，每段开头锁定上一段尾巴，再拼接。

---

## 📄 许可

本项目**免费向所有人开放**使用。作者：**阿里里**。

⭐ 如果这个项目帮到了你，欢迎 Star 与分享！
