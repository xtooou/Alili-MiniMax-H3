"""
MiniMax H3 Pure Sequencer & Continuity Engine (Lightweight Director Core)
剥离了所有二采/脸修冗余，只保留核心：前缀潜空间接力 (Prefix Pinning) + 遮罩去噪 + 分段缓存
"""

from __future__ import annotations
import math
import json
import logging
import torch

log = logging.getLogger("MiniMaxH3.Sequencer")

FPS = 24.0
DEFAULT_CONTEXT_FRAMES = 22  # 官方黄金接力帧数 (约 6 个 latent token)
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
CONTEXT_FRAME_CHOICES = (5, 22, 39, 56)


def snap_context_frames(raw: int | float | None) -> int:
    try:
        n = int(raw or DEFAULT_CONTEXT_FRAMES)
    except (TypeError, ValueError):
        n = DEFAULT_CONTEXT_FRAMES
    return min(CONTEXT_FRAME_CHOICES, key=lambda g: (abs(g - n), -g))


def steps_for_frames(n: int) -> int:
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k


def pixel_frames_for_latent_t(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


MIN_SEGMENT_FRAMES = 124  # H3 训练的最短长度，单段潜空间不要低于它


def plan_segments(total_frames: int, segment_seconds: float, context_frames: int = DEFAULT_CONTEXT_FRAMES) -> list[dict]:
    """把总帧数 (必须是 17n+5) 规划成若干段。

    每段返回 {"length": 该段潜空间对应的帧数(17n+5), "new_frames": 该段净增帧数, "prefix": 借用的前缀帧数}。
    第 1 段没有前缀；之后每段 = 前缀 + 新内容。所有段的总净增帧数恰好等于 total_frames，
    并且各段潜空间长度尽量相等 (显存占用均匀)。
    """
    total = max(5, int(total_frames))
    total += (5 - total) % 17                      # 对齐到 17n+5
    c = snap_context_frames(context_frames)
    p = (c - 5) // 17                              # 前缀占几个 17 帧单元
    units = (total - 5) // 17
    target = max(1.0, float(segment_seconds) * FPS)  # 每段期望的净增帧数
    # 忽略对齐补出来的最多 16 帧，避免 30 秒被多算成 7 段
    n = max(1, math.ceil((total - 16) / target))
    while n > 1:
        v_total = units + p * (n - 1)
        v_min = v_total // n
        if v_min >= p + 1 and 17 * v_min + 5 >= MIN_SEGMENT_FRAMES:
            break
        n -= 1
    if n == 1:
        return [{"length": total, "new_frames": total, "prefix": 0}]
    v_total = units + p * (n - 1)
    base, extra = divmod(v_total, n)
    plan = []
    for i in range(n):
        length = 17 * (base + (1 if i < extra else 0)) + 5
        if i == 0:
            plan.append({"length": length, "new_frames": length, "prefix": 0})
        else:
            plan.append({"length": length, "new_frames": length - c, "prefix": c})
    return plan


def _nested_samples(video: torch.Tensor, audio: torch.Tensor, template=None):
    try:
        from comfy.nested_tensor import NestedTensor
        return NestedTensor((video, audio))
    except Exception:
        cls = type(template) if template is not None else None
        if cls is not None:
            return cls((video, audio))
        raise


def _extract_av_streams(latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError("Expected MiniMax H3 AV NestedTensor")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    return video, audio


def prefix_token_weights(prefix_steps: int, seam_min: float = 0.10) -> tuple[float, ...]:
    """前缀每个 token 的 noise_mask 值。

    约定与 ComfyUI / sampling.py 一致: 0 = 完全保留, 1 = 完全重绘。
    前缀头部全部锁死 (0.0)，最后 4 个 token 从 0 线性升到 seam_min，接缝处轻微重绘。
    """
    n = int(prefix_steps)
    if n < 1:
        return ()
    taper = max(1, min(4, n))
    head = n - taper
    values = [0.0] * head
    values.extend(float(seam_min) * float(i + 1) / float(taper) for i in range(taper))
    return tuple(values)


def apply_h3_latent_continuity(
    target_latent: dict,
    prev_av_latent: dict,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    seam_min: float = 0.10,
) -> tuple[dict, int]:
    """
    核心接力函数：
    把上一段末尾的潜空间粘贴到当前段开头，并打上噪声遮罩 (noise_mask)，强迫当前段顺承上一段动作。
    返回: (打好补丁的 latent, 需要在出图后切除的前缀帧数)
    """
    n_frames = snap_context_frames(context_frames)
    steps = steps_for_frames(n_frames)

    t_video, t_audio = _extract_av_streams(target_latent)
    p_video, p_audio = _extract_av_streams(prev_av_latent)

    total_p_steps = int(p_video.shape[2])
    start_step = max(0, total_p_steps - steps)

    # 1. 提取上一段的视频尾巴
    tail_video = p_video[:, :, start_step:].clone()
    t_tail = min(int(tail_video.shape[2]), int(t_video.shape[2]) - 1)

    patched_video = t_video.clone()
    patched_video[:, :, :t_tail] = tail_video[:, :, :t_tail].to(device=patched_video.device, dtype=patched_video.dtype)

    # 2. 提取音频尾巴
    patched_audio = t_audio.clone()
    total_a_steps = int(p_audio.shape[-1])
    audio_pin_t = min(int(round(n_frames / 24.0 * 40.0)), total_a_steps, int(patched_audio.shape[-1]))
    if audio_pin_t > 0:
        patched_audio[..., :audio_pin_t] = p_audio[..., total_a_steps - audio_pin_t :].to(
            device=patched_audio.device, dtype=patched_audio.dtype
        )

    # 3. 构造 Spatial Noise Mask 遮罩 (前缀锁定，后续完全去噪)
    # mask 与潜空间同形状，和 core.replace_audio_latent 的做法一致
    video_mask = torch.ones_like(patched_video, dtype=torch.float32)
    weights = torch.tensor(prefix_token_weights(t_tail, seam_min), device=patched_video.device, dtype=torch.float32)
    video_mask[:, :, :t_tail] = weights.view(1, 1, t_tail, 1, 1)

    audio_mask = torch.ones_like(patched_audio, dtype=torch.float32)
    audio_mask[..., :audio_pin_t] = 0.0

    out = dict(target_latent)
    tpl = target_latent.get("samples")
    out["samples"] = _nested_samples(patched_video, patched_audio, tpl)
    out["noise_mask"] = _nested_samples(video_mask, audio_mask, tpl)

    trim_pixels = pixel_frames_for_latent_t(t_tail)
    log.info(f"H3 Sequencer: 接力前缀锁定 {t_tail} 个 token ({trim_pixels} 帧)，后续自由生成。")
    return out, trim_pixels


def trim_decoded_prefix(frames: torch.Tensor, audio: dict | None, trim_frames: int, fps: float = FPS) -> tuple[torch.Tensor, dict | None]:
    """出图后一刀切掉前缀借来的帧数，并把音频长度严格对齐到画面时长 (多退少补)"""
    if trim_frames > 0 and int(frames.shape[0]) > trim_frames:
        frames = frames[trim_frames:]
    if isinstance(audio, dict) and audio.get("waveform") is not None:
        sr = int(audio.get("sample_rate") or 32000)
        wave = audio["waveform"]
        drop = int(round((trim_frames / fps) * sr)) if trim_frames > 0 else 0
        if drop > 0 and int(wave.shape[-1]) > drop:
            wave = wave[..., drop:]
        want = int(round((int(frames.shape[0]) / fps) * sr))
        have = int(wave.shape[-1])
        if have > want:
            wave = wave[..., :want]
        elif have < want:
            wave = torch.nn.functional.pad(wave, (0, want - have))
        audio = {"waveform": wave, "sample_rate": sr}
    return frames, audio


def concat_shots(shot_frames: list[torch.Tensor], shot_audios: list[dict | None]) -> tuple[torch.Tensor, dict | None]:
    """无缝拼合所有镜头成片，并平滑音频"""
    if not shot_frames:
        raise ValueError("No frames to concatenate")
    final_frames = torch.cat(shot_frames, dim=0)

    final_audio = None
    valid_audios = [a for a in shot_audios if isinstance(a, dict) and a.get("waveform") is not None]
    if valid_audios:
        sr = valid_audios[0]["sample_rate"]
        waves = [a["waveform"] for a in valid_audios]
        final_audio = {"waveform": torch.cat(waves, dim=-1), "sample_rate": sr}

    return final_frames, final_audio