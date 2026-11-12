from __future__ import annotations

import gc
import math
import os
import re
from enum import Enum

import folder_paths
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

# ===== 原 latent_upscaler_core.py：放大模型定义与加载 =====

MODEL_FOLDER = "latent_upscale_models"
if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER)
    )

VAE_DOWNSAMPLE = 16
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


class UpscaleMode(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"
    MEGAPIXELS = "megapixels"


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _is_rocm_build():
    return getattr(torch.version, "hip", None) is not None


def _resolve_device(backend):
    if backend == "cpu":
        return torch.device("cpu")
    if backend == "rocm":
        if not _is_rocm_build() or not torch.cuda.is_available():
            raise RuntimeError("ROCm was selected, but this PyTorch build cannot access an AMD GPU.")
        return torch.device("cuda")
    if backend == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device backend: {backend}")


def _backend_label(device):
    if device.type == "cuda" and _is_rocm_build():
        return f"ROCm/HIP {torch.version.hip}"
    if device.type == "cuda":
        return f"CUDA {getattr(torch.version, 'cuda', None) or 'unknown'}"
    return "CPU"


def _normalization(channels):
    return nn.GroupNorm(32, channels)


def _zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = _normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, t, h, w = x.shape
        normed = self.norm(x)
        q = self.q(normed).reshape(b, c, t * h * w).transpose(1, 2).unsqueeze(1)
        k = self.k(normed).reshape(b, c, t * h * w).transpose(1, 2).unsqueeze(1)
        v = self.v(normed).reshape(b, c, t * h * w).transpose(1, 2).unsqueeze(1)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.squeeze(1).transpose(1, 2).reshape(b, c, t, h, w)
        return x + self.proj_out(attended)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            _zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        hidden = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(hidden.dtype)
        while emb_out.ndim < hidden.ndim:
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        hidden = self.out_norm(hidden) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(hidden)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0), groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.in_blocks = self._make_blocks(
            in_blocks, channels, embed_dim, dropout, attn,
            temporal_every, temporal_kernel,
        )
        self.out_blocks = self._make_blocks(
            out_blocks, channels, embed_dim, dropout, attn,
            temporal_every, temporal_kernel,
        )
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    @staticmethod
    def _make_blocks(count, channels, embed_dim, dropout, attn,
                     temporal_every, temporal_kernel):
        blocks = nn.ModuleList()
        for index in range(count):
            if (index == 1 or index == count - 1) and attn:
                blocks.append(AttnBlock3D(channels))
            blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and index % temporal_every == 0:
                blocks.append(TemporalConv(channels, temporal_kernel))
        return blocks

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(value * scale)) for value in x.shape[-3:])
        else:
            return x
        if size == x.shape[-3:]:
            return x
        b, c, t, _, _ = x.shape
        temporal_kernel = 0
        for block in self.in_blocks:
            if isinstance(block, TemporalConv):
                temporal_kernel = block.dwconv.weight.shape[2]
                break
        overlap = temporal_kernel
        chunk = 32
        if not enable_chunking or t <= chunk:
            return self._forward_seg(x, scale, size)
        print(f"[MinimaxH3-3D] temporal chunking: T={t} chunks={(t + chunk - 1) // chunk} overlap={overlap}")
        padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        output = torch.zeros(b, c, t, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weights = torch.zeros(1, 1, t, 1, 1, device=x.device, dtype=x.dtype)
        start = 0
        while start < t:
            seg_start = start
            seg_end = min(t, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(t, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(t + 2 * overlap, out_end + overlap)
            segment = padded[:, :, lo:hi].contiguous()
            segment_out = self._forward_seg(segment, scale, (hi - lo, size[-2], size[-1]))
            crop_start = out_start + overlap - lo
            valid = segment_out[:, :, crop_start:crop_start + out_end - out_start]
            weight = torch.ones(out_end - out_start, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                length = seg_start - out_start
                weight[:length] = torch.arange(1, length + 1, device=x.device, dtype=x.dtype) / (length + 1)
            if out_end > seg_end:
                length = out_end - seg_end
                weight[-length:] = torch.arange(length, 0, -1, device=x.device, dtype=x.dtype) / (length + 1)
            weight_view = weight.view(1, 1, -1, 1, 1)
            output[:, :, out_start:out_end] += valid * weight_view
            weights[:, :, out_start:out_end] += weight_view
            start += chunk
            del segment, segment_out, valid
            if start % (chunk * 4) == 0:
                gc.collect()
        return output / weights.clamp_min(1e-8)

    def _forward_seg(self, x, scale, size):
        scale_embedding = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device,
        ).unsqueeze(0)
        embedding = self.embed(scale_embedding)
        x = self.conv_in(x)
        for block in self.in_blocks:
            x = block(x, embedding.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            x = block(x, embedding.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


MODEL_CACHE = {}


def get_models_dir():
    return folder_paths.get_folder_paths(MODEL_FOLDER)[0]


def scan_models():
    names = [
        name for name in folder_paths.get_filename_list(MODEL_FOLDER)
        if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")
    ]
    return names if names else [f"(place models in: {get_models_dir()})"]


def _load_raw_sd(path):
    if path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as file:
                state = {key: file.get_tensor(key) for key in file.keys()}
        except ImportError:
            from safetensors.torch import load_file
            state = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return {
        key: value.to(torch.float16) if value.dtype == torch.float8_e4m3fn else value
        for key, value in state.items()
    }


def _extract_upscaler_sd(state):
    if any(key.startswith("upscaler.") for key in state):
        return {
            key[len("upscaler."):]: value
            for key, value in state.items() if key.startswith("upscaler.")
        }
    return state


def _detect_arch(state):
    config = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12,
        "channels": 512, "dropout": 0.1, "attn": False,
        "temporal_every": 2, "temporal_kernel": 5,
    }
    if "conv_in.weight" in state:
        config["in_channels"] = state["conv_in.weight"].shape[1]
        config["channels"] = state["conv_in.weight"].shape[0]
    input_ids, output_ids = set(), set()
    temporal = False
    for key, value in state.items():
        match = re.match(r"in_blocks\.(\d+)\.in_layers\.", key)
        if match:
            input_ids.add(int(match.group(1)))
        match = re.match(r"out_blocks\.(\d+)\.in_layers\.", key)
        if match:
            output_ids.add(int(match.group(1)))
        if key.endswith("dwconv.weight"):
            temporal = True
            config["temporal_kernel"] = value.shape[2]
    if input_ids:
        config["in_blocks"] = len(input_ids)
    if output_ids:
        config["out_blocks"] = len(output_ids)
    if not temporal:
        config["temporal_every"] = 0
    config["attn"] = False
    return config


def load_model(name, device, precision):
    if str(name).startswith("("):
        raise ValueError("Please place the upscale model in models/latent_upscale_models")
    backend = _backend_label(device)
    cache_key = f"{name}::{backend}::{precision}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key].to(device, non_blocking=True)
    try:
        path = folder_paths.get_full_path_or_raise(MODEL_FOLDER, name)
    except Exception as exc:
        raise FileNotFoundError(f"Model file not found: {name}") from exc
    state = _extract_upscaler_sd(_load_raw_sd(path))
    config = _detect_arch(state)
    model = LatentResizer3D(**config)
    model.load_state_dict(state, strict=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)
    MODEL_CACHE[cache_key] = model
    print(f"[MinimaxH3-3D GH] Loaded upscale model: {name}")
    print(
        f"  Params: {sum(parameter.numel() for parameter in model.parameters()):,} | "
        f"Temporal: {'on' if config['temporal_every'] > 0 else 'off'} "
        f"(every={config['temporal_every']}, kernel={config['temporal_kernel']}) | "
        f"Backend: {backend} | Precision: {precision}"
    )
    return model


# ===== 原 latent_upscaler.py：Latent Upscaler 节点 =====

def _parts(samples):
    if not getattr(samples, "is_nested", False):
        raise ValueError("Minimax H3 GH Upscaler requires AV latent (video + audio)")
    p = tuple(samples.unbind())
    if len(p) != 2 or p[0].ndim != 5 or p[1].ndim != 4:
        raise ValueError("Expected NestedTensor(video[5D], audio[4D])")
    return p[0], p[1]


def _starts(size, tile, overlap):
    if size <= tile:
        return [0]
    step = max(1, tile - overlap)
    out = list(range(0, max(1, size - tile + 1), step))
    last = size - tile
    if out[-1] != last:
        out.append(last)
    return sorted(set(out))


def _window(n, left, right, device):
    w = torch.ones(n, device=device, dtype=torch.float32)
    if left:
        w[:left] = torch.linspace(0.0, 1.0, left + 1, device=device)[1:]
    if right:
        w[-right:] = torch.linspace(1.0, 0.0, right + 1, device=device)[1:]
    return w


class AliliMinimaxH3LatentUpscaler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AliliMinimaxH3LatentUpscalerGH",
            display_name="Minimax H3 Latent Upscaler",
            category="Alili/MiniMax H3 Integration/Upscaling",
            inputs=[
                io.Latent.Input("av_latent"),
                io.Combo.Input("model_name", options=scan_models()),
                io.DynamicCombo.Input("mode", options=[
                    io.DynamicCombo.Option(UpscaleMode.SCALE_BY, [
                        io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05),
                    ]),
                    io.DynamicCombo.Option(UpscaleMode.TARGET_DIMENSIONS, [
                        io.Int.Input("width", default=1280, min=64, max=8192, step=8),
                        io.Int.Input("height", default=704, min=64, max=8192, step=8),
                    ]),
                    io.DynamicCombo.Option(UpscaleMode.MEGAPIXELS, [
                        io.Float.Input("megapixels", default=1.0, min=0.1, max=16.0, step=0.1),
                    ]),
                ]),
                io.Int.Input("align", default=32, min=32, max=512, step=32),
                io.Boolean.Input("force_unload", default=True),
                io.Combo.Input("device", options=["cuda", "rocm", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp16", "bf16", "fp32"], default="fp16"),
            ],
            outputs=[io.Latent.Output("av_latent")],
        )

    @classmethod
    def execute(cls, av_latent, model_name, mode, align, force_unload, device, precision):
        video, audio = _parts(av_latent["samples"])
        orig_dtype = video.dtype
        dev = _resolve_device(device)
        compute_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        s = video.to(dev, dtype=compute_dtype, copy=True)
        b, c, t, h, w = s.shape
        selected = mode["mode"] if isinstance(mode, dict) else str(mode)
        if selected == UpscaleMode.SCALE_BY:
            scale = float(mode["scale"])
            target_w, target_h = w * scale, h * scale
        elif selected == UpscaleMode.TARGET_DIMENSIONS:
            target_w, target_h = float(mode["width"]) / 16, float(mode["height"]) / 16
            scale = ((target_w / w) + (target_h / h)) / 2
        elif selected == UpscaleMode.MEGAPIXELS:
            px = float(mode["megapixels"]) * 1024 * 1024 / (VAE_DOWNSAMPLE ** 2)
            target_h = math.sqrt(px * h / w); target_w = px / target_h; scale = ((target_w / w) + (target_h / h)) / 2
        else:
            raise ValueError(f"Unsupported upscale mode: {selected}")
        target_h = max(1, int(round(target_h / (align / 16))) * int(align / 16))
        target_w = max(1, int(round(target_w / (align / 16))) * int(align / 16))
        oh, ow = int(target_h), int(target_w)
        if oh < h or ow < w:
            raise ValueError("Upscaler only supports scale >= 1")
        if (oh, ow) == (h, w):
            return io.NodeOutput(av_latent)
        model = load_model(model_name, dev, precision)
        mean, std = _make_norm_tensors(dev, compute_dtype)
        def run_spatial_tiled(tile, overlap):
            rows, cols = _starts(h, tile, overlap), _starts(w, tile, overlap)
            out = torch.zeros((b, c, t, oh, ow), device=dev, dtype=compute_dtype)
            weights = torch.zeros((1, 1, 1, oh, ow), device=dev, dtype=torch.float32)
            with torch.inference_mode():
                for r0 in rows:
                    r1 = min(h, r0 + tile)
                    for c0 in cols:
                        c1 = min(w, c0 + tile)
                        piece = (s[:, :, :, r0:r1, c0:c1] - mean) / std
                        pred = model(piece, scale=scale, target_size=(t, max(1, round((r1-r0) * oh / h)), max(1, round((c1-c0) * ow / w))), enable_chunking=True)
                        pred = pred * std + mean
                        pr0, pc0 = round(r0 * oh / h), round(c0 * ow / w)
                        pr1, pc1 = min(oh, pr0 + pred.shape[-2]), min(ow, pc0 + pred.shape[-1])
                        wl = round(overlap * ow / w) if c0 > 0 else 0; wr = round(overlap * ow / w) if c1 < w else 0
                        wt = round(overlap * oh / h) if r0 > 0 else 0; wb = round(overlap * oh / h) if r1 < h else 0
                        win_h = _window(pred.shape[-2], wt, wb, dev); win_w = _window(pred.shape[-1], wl, wr, dev)
                        win = (win_h[:, None] * win_w[None, :]).view(1, 1, 1, pred.shape[-2], pred.shape[-1])
                        out[:, :, :, pr0:pr1, pc0:pc1] += pred[:, :, :, :pr1-pr0, :pc1-pc0] * win[:, :, :, :pr1-pr0, :pc1-pc0]
                        weights[:, :, :, pr0:pr1, pc0:pc1] += win[:, :, :, :pr1-pr0, :pc1-pc0]
                        del piece, pred, win
                        mm.soft_empty_cache()
            return out / weights.clamp_min(1e-8)

        # Fast path: try a single full-frame inference first. If the GPU cannot
        # hold the peak activations, transparently fall back to smaller tiles.
        try:
            with torch.inference_mode():
                full = (s - mean) / std
                out = model(full, scale=scale, target_size=(t, oh, ow), enable_chunking=True)
                out = out * std + mean
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise
            del full
            mm.soft_empty_cache()
            out = None
            tiled_error = None
            for tile_limit in (32, 24, 16, 8, 4):
                tile = max(4, min(h, w, tile_limit))
                overlap = min(tile - 1, max(1, tile // 8))
                try:
                    out = run_spatial_tiled(tile, overlap)
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as tile_exc:
                    if not isinstance(tile_exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(tile_exc).lower():
                        raise
                    tiled_error = tile_exc
                    mm.soft_empty_cache()
                    gc.collect()
            if out is None:
                raise RuntimeError("Minimax H3 GH upscaler could not fit even the smallest automatic tile") from tiled_error
        finally:
            if 's' in locals():
                del s
        out = out.to("cpu", dtype=orig_dtype)
        result = av_latent.copy()
        result["samples"] = NestedTensor((out, audio.to("cpu")))
        masks = av_latent.get("noise_mask")
        if getattr(masks, "is_nested", False):
            mask_parts = tuple(masks.unbind())
            if len(mask_parts) == 2:
                video_mask, audio_mask = mask_parts
                if isinstance(video_mask, torch.Tensor) and video_mask.ndim == 5:
                    vm = video_mask.to(dev, dtype=torch.float32)
                    vm = F.interpolate(vm, size=(vm.shape[2], oh, ow), mode="trilinear", align_corners=False)
                    video_mask = vm.to("cpu", dtype=mask_parts[0].dtype)
                result["noise_mask"] = NestedTensor((video_mask, audio_mask.to("cpu")))
        if dev.type == "cuda" and force_unload:
            model.to("cpu")
            mm.soft_empty_cache()
        gc.collect()
        return io.NodeOutput(result)
