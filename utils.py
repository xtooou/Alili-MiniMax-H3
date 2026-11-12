"""Alili-Segment-Sampler 专用底层辅助工具模块 (自包含独立版)"""

from __future__ import annotations
import torch
from comfy.nested_tensor import NestedTensor
from .h3_sequencer import steps_for_frames
from comfy_extras.nodes_audio import vae_decode_audio

def nested_av_parts(latent: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """提取音视频潜空间 (支持 NestedTensor / tuple / list / dict)"""
    samples = latent.get("samples") if isinstance(latent, dict) else latent
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
        return parts[0], parts[1] if len(parts) > 1 else None
    if isinstance(samples, (tuple, list)):
        return samples[0], samples[1] if len(samples) > 1 else None
    return samples, None

def split_noise_masks(latent: dict, video=None, audio=None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """提取并分离视频和音频的 noise_mask"""
    if not isinstance(latent, dict):
        return None, None
    mask = latent.get("noise_mask")
    if mask is None:
        return None, None
    if hasattr(mask, "unbind"):
        parts = list(mask.unbind())
        return parts[0], parts[1] if len(parts) > 1 else None
    if isinstance(mask, (tuple, list)):
        return mask[0], mask[1] if len(mask) > 1 else None
    return mask, None

def video_latent_t(frames: int) -> int:
    """计算帧数对应的潜空间时间步数"""
    return steps_for_frames(frames)

def decode_av_latent(av_latent: dict, video_vae, audio_vae) -> tuple[torch.Tensor, dict | None, torch.Tensor, torch.Tensor | None]:
    """分段解码音视频潜空间为可视化图像与音频字典"""
    video, audio = nested_av_parts(av_latent)
    images = video_vae.decode(video)
    if images.ndim == 5:
        images = images[0]

    audio_out = None
    if audio is not None and audio_vae is not None:
        audio_out = vae_decode_audio(audio_vae, {"samples": audio})

    return images, audio_out, video, audio

def _clean_locked_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
    """锁定前缀时的干净潜空间回传补丁"""
    return latent_image

def time_shift_sigma(sigma, from_shift: float, to_shift: float):
    """MiniMax H3 双时钟流匹配调度转换"""
    base = sigma / (from_shift + (1.0 - from_shift) * sigma)
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)

def time_shift_slope(sigma, from_shift: float, to_shift: float):
    """MiniMax H3 双时钟流匹配斜率转换"""
    base = sigma / (from_shift + (1.0 - from_shift) * sigma)
    d_base_d_sigma = from_shift / ((from_shift + (1.0 - from_shift) * sigma) ** 2)
    d_sigma_d_base = to_shift / ((1.0 + (to_shift - 1.0) * base) ** 2)
    return d_sigma_d_base * d_base_d_sigma