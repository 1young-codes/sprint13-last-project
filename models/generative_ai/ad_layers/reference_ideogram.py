"""Run BitPoet's Ideogram 4 reference graph on the standalone model code."""

import json

import torch
from torch import nn
from torch.nn import functional as F


def caption_for_edit(prompt, bbox=(0, 0, 1000, 1000)):
    """Let the ad model edit the whole canvas, or bound it to a supplied region."""
    top, left, bottom, right = bbox
    if not (0 <= top < bottom <= 1000 and 0 <= left < right <= 1000):
        raise ValueError("Ideogram edit box must be [top, left, bottom, right] in 0–1000")
    return json.dumps({
        "compositional_deconstruction": {
            "background": "",
            "elements": [{"type": "obj", "bbox": bbox, "desc": prompt}],
        },
    }, ensure_ascii=False, separators=(",", ":"))


def reference_sigmas(width=1024, height=1024):
    from ideogram4.scheduler import get_schedule_for_resolution

    schedule = get_schedule_for_resolution((height, width), known_mean=0.0, std=1.75)
    sigmas = (1 - schedule(torch.linspace(0, 1, 21, dtype=torch.float64))).flip(0)
    sigmas[-1] = 0
    extended = []
    for sigma, following in zip(sigmas[:-1], sigmas[1:]):
        extended.append(sigma)
        if sigma >= 0.98:
            extended.append((sigma + following) / 2)
    return torch.stack([*extended, sigmas[-1]])


def sample_res_multistep(latent, sigmas, predict_clean):
    """Second-order exponential solver, with Euler at the first and last step."""
    previous_clean = None
    for i, (sigma, following) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        clean = predict_clean(latent, sigma)
        if following == 0:
            return clean
        if previous_clean is None:
            latent = clean + (latent - clean) * (following / sigma)
        else:
            step = (sigma / following).log()
            previous_step = (sigmas[i - 1] / sigma).log()
            phi1 = -torch.expm1(-step) / step
            phi2 = (1 - phi1) / step
            second = phi2 * step / previous_step
            latent = (-step).exp() * latent + step * (phi1 * clean + second * (clean - previous_clean))
        previous_clean = clean
    raise ValueError("Expected a schedule ending at zero")


class ScaledFP8Linear(nn.Module):
    """Read the FP8 tensors and scales from a public repack."""

    compute_dtype = torch.bfloat16

    def __init__(self, base):
        super().__init__()
        self.register_buffer("weight", torch.empty_like(base.weight, dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32, device=base.weight.device))
        self.register_buffer("bias", base.bias)
        self.register_buffer("input_scale", None, persistent=False)
        self.register_buffer("down", None, persistent=False)
        self.register_buffer("up", None, persistent=False)

    def forward(self, x):
        if x.is_cuda and self.down is None and x.ndim in (2, 3):
            flat = x.flatten(0, -2)
            scale = self.input_scale
            if scale is None:
                scale = (flat.abs().amax().float() / 448).clamp_min(1e-12)
            # The Ada kernel works with a multiple of 16 activation rows.
            padded = F.pad(flat, (0, 0, 0, -flat.shape[0] % 16))
            quant = (padded.float() / scale).clamp_(-448, 448).to(self.weight.dtype)
            y = torch._scaled_mm(quant, self.weight.T, scale, self.weight_scale, out_dtype=x.dtype)
            y = y[:flat.shape[0]].reshape(*x.shape[:-1], self.weight.shape[0])
            return y if self.bias is None else y + self.bias
        weight = self.weight.to(dtype=x.dtype).mul_(self.weight_scale)
        prediction = F.linear(x, weight, self.bias)
        if self.down is not None:
            prediction += F.linear(F.linear(x, self.down), self.up)
        return prediction


def load_scaled_weights(model, path, device, prefix=""):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for name, child in list(model.named_modules()):
            if isinstance(child, nn.Linear) and f"{prefix}{name}.weight_scale" in keys:
                parent, _, key = name.rpartition(".")
                linear = ScaledFP8Linear(child)
                if f"{prefix}{name}.input_scale" in keys:
                    linear.input_scale = source.get_tensor(f"{prefix}{name}.input_scale").to(device, torch.float32)
                setattr(model.get_submodule(parent), key, linear)
        state = {}
        for name in model.state_dict():
            tensor = source.get_tensor(prefix + name)
            dtype = tensor.dtype if name.endswith(".weight_scale") or tensor.dtype == torch.float8_e4m3fn else torch.bfloat16
            state[name] = tensor.to(device=device, dtype=dtype)
        model.load_state_dict(state, assign=True)
    return model.requires_grad_(False).eval()


def load_edit_lora(model, path, device):
    from safetensors import safe_open

    start, end = "diffusion_model.", ".lora_A.weight"
    with safe_open(path, framework="pt", device="cpu") as source:
        for key in source.keys():
            if not key.endswith(end):
                continue
            name = key.removeprefix(start).removesuffix(end)
            layer = model.get_submodule(name)
            if not isinstance(layer, ScaledFP8Linear):
                raise TypeError(f"Unexpected Ideogram LoRA target {name}")
            layer.down = source.get_tensor(key).to(device, torch.bfloat16)
            layer.up = source.get_tensor(f"{start}{name}.lora_B.weight").to(device, torch.bfloat16)


def encode_prompts(jobs, weights, device, edit_box):
    from ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding

    config = AutoConfig.from_pretrained(weights / "tokenizer", local_files_only=True).text_config
    tokenizer = AutoTokenizer.from_pretrained(weights / "tokenizer", local_files_only=True)
    with torch.device("meta"):
        encoder = Qwen3VLTextModel(config)
    encoder.rotary_emb = Qwen3VLTextRotaryEmbedding(config).to(device)
    load_scaled_weights(encoder, weights / "text_encoders/qwen3vl_8b_fp8_scaled.safetensors", device, "model.")
    with torch.inference_mode():
        results = []
        for prompt, _ in jobs:
            caption = caption_for_edit(prompt, edit_box)
            prompt = f"<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n"
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            captured = {}
            hooks = [
                encoder.layers[index].register_forward_hook(
                    lambda _module, _args, value, tap=index: captured.__setitem__(tap, value),
                )
                for index in QWEN3_VL_ACTIVATION_LAYERS
            ]
            encoder(tokens, use_cache=False)
            for hook in hooks:
                hook.remove()
            results.append(torch.stack([captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS], dim=-1).flatten(2).cpu())
    return results


def predict_velocity(model, target, time, positions, text=None, reference=None):
    # The released model uses the same image embedding for the target and the reference.
    image = target if reference is None else torch.cat((target, reference), dim=1)
    hidden = model.input_proj(image.to(torch.bfloat16))
    hidden += model.embed_image_indicator.weight[1]
    if text is not None:
        text_hidden = model.llm_cond_proj(model.llm_cond_norm(text)) + model.embed_image_indicator.weight[0]
        hidden = torch.cat((text_hidden, hidden), dim=1)
    if reference is not None:
        times = time.expand(1, hidden.shape[1]).clone()
        times[:, -reference.shape[1]:] = 1.0
        time = times
    else:
        time = time.reshape(1, 1)
    modulation = F.silu(model.adaln_proj(model.t_embedding(time.to(torch.bfloat16))))
    cos, sin = model.rotary_emb(positions)
    cos, sin = cos.to(hidden.dtype), sin.to(hidden.dtype)
    for layer in model.layers:
        hidden = layer(hidden, None, cos, sin, modulation)
    start = 0 if text is None else text.shape[1]
    end = start + target.shape[1]
    if reference is not None:
        modulation = modulation[:, start:end]
    return model.final_layer(hidden[:, start:end], modulation).float()


def generate_batch(jobs, weights, device="cuda", edit_box=(0, 0, 1000, 1000)):
    from flux3.autoencoder import AutoEncoder, latent_ids
    from flux3.checkpoint import load_checkpoint
    from flux3.pipeline import to_pil, to_tensor
    from ideogram4.constants import IMAGE_POSITION_OFFSET
    from ideogram4.modeling_ideogram4 import (
        Ideogram4Attention, Ideogram4Config, Ideogram4MRoPE, Ideogram4Transformer, _apply_rotary_pos_emb,
    )

    home = weights / "ideo"
    print("Encoding Ideogram prompts", flush=True)
    texts = encode_prompts(jobs, home, device, edit_box)
    torch.cuda.empty_cache()
    vae = load_checkpoint(AutoEncoder(), str(home / "vae/flux2-vae.safetensors"), device, torch.float32).eval()
    size = 1024
    grid = size // 16
    ids = latent_ids(grid, grid, device)
    image_positions = ids[None, :, :3] + IMAGE_POSITION_OFFSET
    ref_positions = image_positions.clone()
    ref_positions[..., 0] += 1
    schedule = reference_sigmas().to(device)
    config = Ideogram4Config()

    class SingleImageAttention(Ideogram4Attention):
        def forward(self, x, segment_ids, cos, sin):
            # A single unpadded sequence has no segment boundaries.
            batch, length, channels = x.shape
            q, k, v = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim).unbind(2)
            q = self.norm_q(q).transpose(1, 2)
            k = self.norm_k(k).transpose(1, 2)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)
            out = F.scaled_dot_product_attention(q, k, v.transpose(1, 2))
            return self.o(out.transpose(1, 2).reshape(batch, length, channels))

    def load_transformer(filename, placement):
        with torch.device("meta"):
            model = Ideogram4Transformer(config)
            for layer in model.layers:
                layer.attention = SingleImageAttention(config.emb_dim, config.num_heads, config.norm_eps)
        model.rotary_emb = Ideogram4MRoPE(
            config.emb_dim // config.num_heads, config.rope_theta, config.mrope_section,
        ).to(placement)
        return load_scaled_weights(model, home / f"diffusion_models/{filename}.safetensors", placement)

    print("Loading Ideogram denoisers", flush=True)
    positive = load_transformer("ideogram4_fp8_scaled", device)
    load_edit_lora(positive, weights / "ideo_lora/V3/Ido4Inpaint_V3_1k_000006400.safetensors", device)
    negative = load_transformer("ideogram4_unconditional_fp8_scaled", "cpu")
    for text, (_, source) in zip(texts, jobs, strict=True):
        with torch.inference_mode():
            rgb = F.interpolate(
                to_tensor(source).unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False,
            )[0].to(device)
            packed, _ = vae.encode(rgb)
            reference = packed.unflatten(-1, (32, 4)).transpose(-2, -1).reshape(1, -1, 128)
            text = text.to(device)
            negative_text = torch.zeros_like(text)
            text_positions = torch.arange(text.shape[1], device=device).reshape(1, -1, 1).expand(-1, -1, 3)
            negative_positions = torch.cat((text_positions, image_positions), dim=1)
            positions = torch.cat((negative_positions, ref_positions), dim=1)
            noise = torch.randn((1, 128, grid, grid), generator=torch.Generator().manual_seed(1)).to(device)
            latent = noise[0].flatten(1).T.unflatten(-1, (32, 4)).transpose(-2, -1).reshape(1, -1, 128)
            negative.to(device)

            def predict_clean(sample, sigma):
                time = 1 - sigma
                velocity = predict_velocity(positive, sample, time, positions, text, reference)
                if sigma > 0.1:
                    uncond = predict_velocity(negative, sample, time, negative_positions, negative_text)
                    velocity = uncond + 3 * (velocity - uncond)
                return sample + sigma * velocity

            latent = sample_res_multistep(latent * schedule[0], schedule, predict_clean)
            if not torch.isfinite(latent).all():
                raise RuntimeError("Ideogram returned non-finite latents")
            packed = latent[0].unflatten(-1, (4, 32)).transpose(-2, -1).reshape(-1, 128)
            # The full-size VAE decode needs memory beyond both resident denoisers.
            negative.to("cpu")
            torch.cuda.empty_cache()
            image = to_pil(vae.decode(packed, ids))
        yield image
