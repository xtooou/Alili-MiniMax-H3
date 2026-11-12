"""MiniMax H3 自动分段采样节点 (Alili 独立版，支持首采/二采与双路降噪输出)"""

from __future__ import annotations

import copy
import gc
import logging
import math
import types
import uuid

import torch
import comfy.model_management as model_management
import comfy.conds
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

from .h3_sequencer import (
    DEFAULT_CONTEXT_FRAMES,
    apply_h3_latent_continuity,
    concat_shots,
    pixel_frames_for_latent_t,
    plan_segments,
    snap_context_frames,
    steps_for_frames,
    trim_decoded_prefix,
)
from .tiled_sampler import _AliliMinimaxH3TiledSamplerLegacy
from .utils import (
    _clean_locked_latent_inpaint,
    decode_av_latent,
    nested_av_parts,
    split_noise_masks,
    video_latent_t,
)

log = logging.getLogger("Alili.AutoSegment")

NODE_CATEGORY = "Alili/MiniMax H3 Integration"
MODE_GENERATE = "首采(生成)"
MODE_REFINE = "二采(修复放大)"


def _noise_for_segment(noise, seed: int):
    """复制 noise 对象并换种子，保证每段噪声不同。"""
    seg_noise = copy.copy(noise)
    seg_noise.seed = int(seed)
    return seg_noise


def _window_reference_audio(refs, tick_a: int, tick_n: int, total_ticks: int):
    """原声锁定时，把「整段长度」的参考音频切成和本段窗口对应的那一小段。"""
    out = []
    for ref in refs:
        latent = ref.get("audio_latent") if isinstance(ref, dict) else None
        if (
            isinstance(ref, dict) and ref.get("kind") == "audio" and torch.is_tensor(latent)
            and abs(int(latent.shape[-1]) - total_ticks) <= 4
        ):
            piece = latent[..., tick_a:tick_a + tick_n]
            if int(piece.shape[-1]) < tick_n:
                piece = torch.cat((piece, piece.new_zeros((*piece.shape[:-1], tick_n - int(piece.shape[-1])))), dim=-1)
            ref = dict(ref)
            ref["audio_latent"] = piece.clone()
            ref["ref_audio_t"] = int(piece.shape[-1])
        out.append(ref)
    return out
    
    
def _find_segment_conds(guider):
    """找出适配器挂在条件里的分镜条件列表，没有就返回 None。"""
    for conds in getattr(guider, "original_conds", {}).values():
        for cond in conds:
            if isinstance(cond, dict) and cond.get("alili_segment_conds"):
                return cond["alili_segment_conds"]
    return None    


def _segment_guider(guider, index: int, count: int, frame_count: int, clean_patch: bool, state: dict,
                    audio_window: tuple[int, int, int] | None = None):
    """给第 index 段克隆一个 guider，并改写条件里的首尾帧。"""
    seg = copy.copy(guider)
    new_conds = {}
    for key, conds in getattr(guider, "original_conds", {}).items():
        rewritten = []
        for j, cond in enumerate(conds):
            if not isinstance(cond, dict):
                rewritten.append(cond)
                continue
            cond = dict(cond)
            cond["uuid"] = uuid.uuid4()
            stash = cond.pop("alili_segment_conds", None)
            if stash is not None:
                src_t, src_d = stash[index][j]
                for k, v in src_d.items():
                    if k not in ("model_conds", "uuid") and not k.startswith("minimax_"):
                        cond[k] = v
                cond["cross_attn"] = src_t
                mc = dict(cond.get("model_conds", {}))
                mc["c_crossattn"] = comfy.conds.CONDCrossAttn(src_t)
                cond["model_conds"] = mc
                log.info("Alili AutoSegment: 第 %d 段使用独立文本条件，形状 %s", index + 1, tuple(src_t.shape))            
            keyframes = cond.get("minimax_keyframes")
            if keyframes:
                kept = []
                for keyframe in keyframes:
                    is_first = int(keyframe.get("resolved_frame_index", 0)) == 0
                    if is_first and index == 0:
                        kept.append(dict(keyframe))
                    elif not is_first and index == count - 1:
                        last = dict(keyframe)
                        last["resolved_frame_index"] = frame_count - 1
                        kept.append(last)
                if kept:
                    cond["minimax_keyframes"] = kept
                    cond["minimax_frame_count"] = frame_count
                else:
                    cond.pop("minimax_keyframes", None)
                    cond.pop("minimax_frame_count", None)
            if audio_window is not None and cond.get("minimax_refs"):
                cond["minimax_refs"] = _window_reference_audio(cond["minimax_refs"], *audio_window)
            rewritten.append(cond)
        new_conds[key] = rewritten
    seg.original_conds = new_conds

    if clean_patch:
        try:
            if "patcher" not in state:
                patcher = guider.model_patcher.clone()
                patcher.add_object_patch(
                    "scale_latent_inpaint",
                    types.MethodType(_clean_locked_latent_inpaint, patcher.model),
                )
                state["patcher"] = patcher
            seg.model_patcher = state["patcher"]
            seg.model_options = state["patcher"].model_options
        except Exception as exc:
            log.warning("Alili AutoSegment: clean prefix patch skipped (%s)", exc)
    return seg


def _ticks(frames: int) -> int:
    """视频帧数 -> 音频潜空间 tick 数 (24fps -> 40Hz，四舍五入)。"""
    return int(math.floor(frames * 40.0 / 24.0 + 0.5))


def _layout(plan: list[dict], total_frames: int) -> list[dict]:
    """每段在总时间线上的位置。"""
    out, cursor = [], 0
    for seg in plan:
        start, end = cursor, cursor + seg["new_frames"]
        out.append({
            "window_start": start - seg["prefix"],
            "start": start,
            "end": end,
            "ticks_keep": _ticks(end) - _ticks(start),
        })
        cursor = end
    if cursor != total_frames:
        raise ValueError(f"分段规划总帧数 {cursor} 与潜空间 {total_frames} 不一致")
    return out


def _slice_dim(mask, start: int, stop: int, dim: int, full: int):
    """时间维和潜空间等长才切片。"""
    if mask is None or mask.shape[dim] != full:
        return mask
    index = [slice(None)] * mask.ndim
    index[dim] = slice(start, stop)
    return mask[tuple(index)]


def _slice_window(latent: dict, window_start: int, length: int) -> dict:
    """从完整潜空间里取出一段窗口。"""
    if window_start % 17:
        raise ValueError(f"窗口起点 {window_start} 不在 17 帧网格上")
    video, audio = nested_av_parts(latent)
    step_a = window_start // 17 * 5
    step_n = video_latent_t(length)
    tick_a, tick_n = _ticks(window_start), _ticks(length)
    if step_a + step_n > int(video.shape[2]):
        raise ValueError("分段窗口超出视频潜空间长度")

    seg_video = video[:, :, step_a:step_a + step_n].clone()
    seg_audio = audio[..., tick_a:tick_a + tick_n].clone()
    if int(seg_audio.shape[-1]) < tick_n:
        pad = seg_audio.new_zeros((*seg_audio.shape[:-1], tick_n - int(seg_audio.shape[-1])))
        seg_audio = torch.cat((seg_audio, pad), dim=-1)

    out = {k: v for k, v in latent.items() if k not in ("samples", "noise_mask")}
    out["samples"] = NestedTensor((seg_video, seg_audio))

    video_mask, audio_mask = split_noise_masks(latent, video, audio)
    if video_mask is not None:
        video_mask = _slice_dim(video_mask, step_a, step_a + step_n, 2, int(video.shape[2]))
        if audio_mask is None:
            audio_mask = torch.ones_like(seg_audio, dtype=torch.float32)
        else:
            audio_mask = _slice_dim(audio_mask, tick_a, tick_a + tick_n, audio_mask.ndim - 1, int(audio.shape[-1]))
            if int(audio_mask.shape[-1]) not in (1, int(seg_audio.shape[-1])):
                audio_mask = torch.ones_like(seg_audio, dtype=torch.float32)
        out["noise_mask"] = NestedTensor((video_mask.clone(), audio_mask.clone()))
    return out


def _merge_masks(continued: dict, sliced: dict) -> dict:
    """合并原有 mask 与接力 mask。"""
    if "noise_mask" not in sliced:
        return continued
    cv, ca = continued["noise_mask"].unbind()
    sv, sa = sliced["noise_mask"].unbind()
    merged_v = torch.minimum(cv, sv.to(device=cv.device, dtype=cv.dtype))
    merged_a = torch.minimum(ca, sa.to(device=ca.device, dtype=ca.dtype))
    out = dict(continued)
    out["noise_mask"] = NestedTensor((merged_v, merged_a))
    return out


class AliliMiniMaxH3AutoSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AliliMiniMaxH3AutoSegment",
            display_name="Alili H3 自动分段采样器 (首采/二采)",
            description=(
                "把长视频拆成若干短段依次采样或修复：每段开头锁定上一段的尾巴，"
                "解码后切掉借来的前缀再拼接。首采和二采通用，同时输出普通潜空间与无噪降噪潜空间。"
            ),
            category=NODE_CATEGORY,
            inputs=[
                io.Combo.Input(
                    "pass_mode", options=[MODE_GENERATE, MODE_REFINE], default=MODE_GENERATE,
                    display_name="采样阶段",
                    tooltip="首采(生成)：按输入潜空间的尺寸和时长从空白生成，口型驱动音频会自动按段对应。\n"
                            "二采(修复放大)：对输入潜空间里已有的内容逐段重绘，配合部分去噪的 Sigmas 使用。",
                ),
                io.Noise.Input("noise", display_name="噪波生成"),
                io.Guider.Input("guider", display_name="引导"),
                io.Sampler.Input("sampler", display_name="采样器"),
                io.Sigmas.Input("sigmas", display_name="Sigmas"),
                io.Latent.Input(
                    "latent_image", display_name="输入潜空间",
                    tooltip="首采接适配器的「音视频潜空间」；二采接放大后的潜空间。总时长由它决定。",
                ),
                io.Vae.Input("video_vae", display_name="视频 VAE"),
                io.Vae.Input("audio_vae", display_name="音频 VAE"),
                io.Float.Input(
                    "segment_seconds", default=5.0, min=4.0, max=20.0, step=0.5,
                    display_name="每段秒数",
                    tooltip="每段净增的秒数，按显存一次能跑的长度设置。每段实际还要多出前缀约 0.9 秒。",
                ),
                io.Combo.Input(
                    "context_frames", options=["5", "22", "39", "56"], default=str(DEFAULT_CONTEXT_FRAMES),
                    display_name="接力帧数",
                    tooltip="每段开头借用上一段末尾的帧数。接缝有抖动时可调大。",
                ),
                io.Float.Input(
                    "seam_redraw", default=0.10, min=0.0, max=0.5, step=0.01,
                    display_name="接缝重绘强度",
                    tooltip="前缀最后几帧允许被轻微重绘的强度，0 为完全锁死。",
                ),
                io.Boolean.Input(
                    "clean_prefix_condition", default=True,
                    display_name="干净前缀条件",
                    tooltip="把锁定的前缀以干净潜空间喂给模型 (与原声锁定的处理方式一致)。接缝有伪影时可关闭对比。",
                ),
                io.Boolean.Input(
                    "decode_output", default=True,
                    display_name="分段解码输出画面",
                    tooltip="逐段解码并输出画面和音频。后面还要接放大或二采时请关闭，省时间和内存。",
                ),
                io.Boolean.Input(
                    "enable_tiling", default=False, display_name="启用画面分块",
                    tooltip="每段内部按画面空间分块采样 (同二采分块采样器)",
                ),
                io.Int.Input("n_tiles", default=2, min=1, max=8, display_name="分块数"),
                io.Int.Input("tile_overlap", default=128, min=0, max=2048, step=64, display_name="分块重叠(像素)"),
                io.Boolean.Input("refine_seams", default=False, display_name="修补分块接缝"),
                io.Int.Input("refine_steps", default=4, min=1, max=25, display_name="接缝修补步数"),
            ],
            outputs=[
                io.Latent.Output("latent", display_name="潜空间"),
                io.Latent.Output("denoised_latent", display_name="降噪输出"),
                io.Image.Output("frames", display_name="图像"),
                io.Audio.Output("audio", display_name="音频"),
                io.String.Output("report", display_name="分段报告"),
            ],
        )

    @classmethod
    def execute(cls, pass_mode, noise, guider, sampler, sigmas, latent_image, video_vae, audio_vae,
                segment_seconds, context_frames, seam_redraw, clean_prefix_condition, decode_output,
                enable_tiling, n_tiles, tile_overlap, refine_seams, refine_steps):
        refine = pass_mode == MODE_REFINE

        video, audio_stream = nested_av_parts(latent_image)
        _, input_audio_mask = split_noise_masks(latent_image, video, audio_stream)
        audio_locked = input_audio_mask is not None and bool((input_audio_mask < 1.0).any().item())
        total_frames = pixel_frames_for_latent_t(int(video.shape[2]))
        height, width = int(video.shape[-2]) * 16, int(video.shape[-1]) * 16
        context = snap_context_frames(int(context_frames))
        plan = plan_segments(total_frames, segment_seconds, context)
        layout = _layout(plan, total_frames)
        seg_conds = _find_segment_conds(guider)
        if seg_conds is not None and len(seg_conds) != len(plan):
            raise ValueError(
                f"分镜数量不匹配：采样器需要 {len(plan)} 段，提示词里拆出 {len(seg_conds)} 个分镜。"
                f"请让 LLM 输出 {len(plan)} 个分镜，或调整每段秒数。"
            )        
        base_seed = int(getattr(noise, "seed", 0))
        total_ticks = _ticks(total_frames)
        report = [
            f"{'二采/修复' if refine else '首采'}：总帧数 {total_frames} ({total_frames / 24.0:.2f}s)，"
            f"分 {len(plan)} 段，前缀 {context} 帧，画面 {width}x{height}，"
            f"原声锁定={'是' if audio_locked else '否'}",
        ]

        sampler_impl = _AliliMinimaxH3TiledSamplerLegacy()
        patch_state: dict = {}
        prev_av = None

        # 双路潜空间碎片收集：一路普通采样，一路降噪无噪预测
        video_chunks, audio_latent_chunks = [], []
        denoised_video_chunks, denoised_audio_latent_chunks = [], []
        frame_chunks, audio_chunks = [], []

        for index, (segment, place) in enumerate(zip(plan, layout)):
            model_management.throw_exception_if_processing_interrupted()

            latent = _slice_window(latent_image, place["window_start"], segment["length"])
            frame_count = segment["length"]

            trim = 0
            if index > 0:
                sliced = latent
                latent, trim = apply_h3_latent_continuity(latent, prev_av, context, seam_redraw)
                latent = _merge_masks(latent, sliced)
                latent["alili_h3_audio_mode"] = "segment_continue"

            seg_guider = _segment_guider(
                guider, index, len(plan), frame_count,
                clean_patch=bool(clean_prefix_condition) and index > 0,
                state=patch_state,
                audio_window=(
                    (_ticks(place["window_start"]), _ticks(segment["length"]), total_ticks) if audio_locked else None
                ),
            )
            seg_noise = _noise_for_segment(noise, base_seed + index)

            log.info("Alili AutoSegment(%s): 第 %d/%d 段，潜空间 %d 帧，净增 %d 帧", pass_mode, index + 1,
                     len(plan), segment["length"], segment["new_frames"])
            
            # 捕获双返回值：sampled (带噪态), denoised (纯净 x0 预测态)
            sampled, denoised = sampler_impl.sample_tiled(
                seg_noise, seg_guider, sampler, sigmas, latent,
                enable_tiling=bool(enable_tiling), n_tiles=int(n_tiles), tile_overlap=int(tile_overlap),
                refine_seams=bool(refine_seams), refine_steps=int(refine_steps),
            )
            if denoised is None:
                denoised = sampled

            # 潜空间拼接：视频去掉前缀那几步，音频取该段对应的 tick 数
            sampled_video, sampled_audio = nested_av_parts(sampled)
            denoised_video, denoised_audio = nested_av_parts(denoised)
            skip_steps = steps_for_frames(trim) if trim else 0

            # 1. 收集普通 sampled 结果
            video_chunks.append(sampled_video[:, :, skip_steps:].clone())
            audio_latent_chunks.append(sampled_audio[..., -place["ticks_keep"]:].clone())

            # 2. 收集纯净 denoised 结果 (解决二采发糊的关键)
            denoised_video_chunks.append(denoised_video[:, :, skip_steps:].clone())
            d_audio = denoised_audio if denoised_audio is not None else sampled_audio
            denoised_audio_latent_chunks.append(d_audio[..., -place["ticks_keep"]:].clone())

            del sampled_video, sampled_audio, denoised_video, denoised_audio

            kept = int(segment["new_frames"])
            if decode_output:
                # 二采模式下优先解码干净的 denoised，确保实时预览也是锐利清晰的
                target_decode = denoised if (refine and denoised is not None) else sampled
                frames, audio, _, _ = decode_av_latent(target_decode, video_vae, audio_vae)
                frames = frames[: trim + kept]
                frames, audio = trim_decoded_prefix(frames, audio, trim)
                frame_chunks.append(frames)
                audio_chunks.append(audio)
                kept = int(frames.shape[0])
                del frames, audio
            report.append(f"第 {index + 1} 段：潜空间 {segment['length']} 帧，切除前缀 {trim} 帧，保留 {kept} 帧")

            # 接力前缀：二采优先使用干净的 denoised 作为下一段的前缀底图，防止把残余噪声污染到下一段
            relay_source = denoised if (refine and denoised is not None) else sampled
            prev_av = {"samples": relay_source["samples"]}
            del sampled, denoised, latent, seg_guider
            gc.collect()
            model_management.soft_empty_cache()

        # 提取原本输入的全局 mask (保留音频锁定状态)
        orig_vm, orig_am = split_noise_masks(latent_image, video, audio_stream)

        # 构造最终普通潜空间
        final_latent = {k: v for k, v in latent_image.items() if k not in ("samples", "noise_mask")}
        final_latent["samples"] = NestedTensor(
            (torch.cat(video_chunks, dim=2), torch.cat(audio_latent_chunks, dim=-1))
        )
        if orig_am is not None:
            # 视频区全部允许二采(1.0)，音频区保留原本的锁定值(0.0)
            vm_full = torch.ones_like(final_latent["samples"].unbind()[0], dtype=torch.float32)
            final_latent["noise_mask"] = NestedTensor((vm_full, orig_am))

        # 构造最终纯净降噪潜空间 (提供给下游二采/高清输出)
        final_denoised_latent = {k: v for k, v in latent_image.items() if k not in ("samples", "noise_mask")}
        final_denoised_latent["samples"] = NestedTensor(
            (torch.cat(denoised_video_chunks, dim=2), torch.cat(denoised_audio_latent_chunks, dim=-1))
        )
        if orig_am is not None:
            vm_denoised = torch.ones_like(final_denoised_latent["samples"].unbind()[0], dtype=torch.float32)
            final_denoised_latent["noise_mask"] = NestedTensor((vm_denoised, orig_am))

        if decode_output:
            final_frames, final_audio = concat_shots(frame_chunks, audio_chunks)
            report.append(f"合计 {int(final_frames.shape[0])} 帧 ({final_frames.shape[0] / 24.0:.2f}s)")
        else:
            final_frames, final_audio = torch.zeros((1, 64, 64, 3)), None
            report.append("decode_output 已关闭：只输出潜空间，画面为占位图")

        return io.NodeOutput(final_latent, final_denoised_latent, final_frames, final_audio, "\n".join(report))