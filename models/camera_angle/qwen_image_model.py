"""Qwen-Image 2.1 이미지 편집 모델을 INT8 가중치로 불러와 실행합니다 (그래픽 카드 필요, ComfyUI 없이 diffusers만 사용).

ai-generation 프로젝트의 backend/studio/engine/qwen21.py와 qwen21_vae.py를 옮겼습니다.
원래 코드는 10GB 그래픽 카드용으로 블록 단위 오프로드를 썼지만, 여기서는 L4(24GB)에 맞춰
생성 모델(약 6.8GB)과 VAE는 그래픽 카드에 올려 두고, 문장 인코더(약 8.8GB)만 쓸 때 올렸다가 내립니다.

샘플링 설정은 원본 예제 작업 흐름과 같습니다: Euler, simple 시그마(shift 0.69), CFG 1, 참고 해상도 0.
"""

import gc
import json
import math
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .int8_linear_layers import (attach_lora, check_int8, check_names, convrot_group, layer_scale,
                                 load_weights_into_empty_model, read_safetensors_header, remove_loras, rotate)
from .settings import load_settings, model_file, model_folder
from .speed_up_generation import StepChangeRecorder, mag_cache_config, register_qwen21_blocks

DEVICE = "cuda"


def aligned_size(width: int, height: int) -> tuple[int, int]:
    """모델이 받는 크기: 가로·세로를 32의 배수로 맞춥니다."""
    return tuple(max(32, round(n / 32) * 32) for n in (width, height))


def simple_sigmas(steps: int, shift: float = .69) -> list[float]:
    t = (10000 - np.floor(np.arange(steps) * (10000 / steps))) / 10000
    return (math.exp(shift) / (math.exp(shift) + 1 / t - 1)).astype(np.float32).tolist()


class FusedMLP(nn.Module):
    """체크포인트의 gate_up 묶음 층 구조에 맞춘 MLP. 메모리를 아끼려고 256토큰씩 나눠 계산합니다."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(width, hidden * 2, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        result = torch.empty_like(x)
        for start in range(0, x.shape[1], 256):
            gate, up = self.gate_up(x[:, start:start + 256]).chunk(2, dim=-1)
            result[:, start:start + 256] = self.out(torch.nn.functional.silu(gate) * up)
        return result


class Int8Embedding(nn.Module):
    """INT8로 저장된 단어 임베딩. 필요한 토큰 행만 실수로 풀어 씁니다."""

    def __init__(self, weight, scale, group):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("scale", scale, persistent=False)
        self.group = group

    def forward(self, ids):
        values = self.weight[ids].float() * self.scale[ids]
        return (rotate(values, self.group) if self.group else values).to(torch.bfloat16)


def whole_frame_vae_class():
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import (AutoencoderKLQwenImage21,
                                                                         QwenImage21RMS_norm)

    def normalize_in_row_bands(normalize, values):
        # 채널 정규화는 픽셀마다 독립이라, 큰 이미지에서는 행 묶음으로 나눠 FP32 중간값 크기를 제한합니다
        if values.numel() <= 96 * 256 ** 2:
            return normalize(values)
        rows = max(1, min(64, (64 * 2880) // values.shape[-1]))
        result = torch.empty_like(values)
        for start in range(0, values.shape[-2], rows):
            band = slice(start, start + rows)
            result[..., band, :] = normalize(values[..., band, :])
        return result

    class WholeFrameVAE(AutoencoderKLQwenImage21):
        """참고 이미지를 조각내지 않고 한 장 전체로 인코딩합니다.
        조각으로 나누면 초록색만 있는 조각은 장면 맥락을 잃습니다."""

        def limit_normalization_memory(self):
            # 인코더와 디코더 모두: 2메가픽셀 한 장을 조각 없이 복원할 때 정규화 임시값이 그래픽 메모리를 넘지 않게 합니다
            for layer in self.modules():
                if isinstance(layer, QwenImage21RMS_norm) and layer.channel_first:
                    layer.forward = partial(normalize_in_row_bands, layer.forward)

        def _encode(self, pixels):
            if pixels.shape[2] != 1 or self.config.patch_size is not None:
                raise ValueError("한 장 전체 인코딩은 단일 프레임 Qwen VAE에서만 됩니다")
            return self.quant_conv(self.encoder(pixels))

    return WholeFrameVAE


def vae_key_names(header: dict) -> dict[str, str]:
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    names = {}
    for key in header:
        match = re.match(r"(encoder\.down|decoder\.up)samples\.(\d+)\.(?:down|up)samples\.(\d+)\.(.*)", key)
        if match:
            side, block, layer, tail = match.groups()
            is_up = side.startswith("decoder")
            if int(layer) == (3 if is_up else 2):
                prefix = f"{side}_blocks.{block}.{'up' if is_up else 'down'}sampler."
            else:
                prefix = f"{side}_blocks.{block}.resnets.{layer}."
            tail = tail.replace("shortcut.", "conv_shortcut.")
            for old, new in (("residual.0.", "norm1."), ("residual.2.", "conv1."),
                             ("residual.3.", "norm2."), ("residual.6.", "conv2.")):
                tail = tail.replace(old, new)
            names[prefix + tail] = key
        else:
            names.update(convert_wan_vae_to_diffusers({key: key}))
    return names


def load_vae(folder: Path):
    from accelerate import init_empty_weights
    from safetensors import safe_open

    vae_class = whole_frame_vae_class()
    with init_empty_weights():
        vae = vae_class.from_config(vae_class.load_config(folder / "vae"))
    vae.limit_normalization_memory()
    path = model_file("qwen_vae")
    header = read_safetensors_header(path)
    names = check_names(vae, header, vae_key_names(header), squeeze_time=True)
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        for name, key in names.items():
            tensor = source.get_tensor(key)
            state[name] = (tensor.squeeze(2) if tensor.ndim == 5 else tensor).to(torch.bfloat16)
    vae.load_state_dict(state, strict=True, assign=True)
    # 조각(타일) 복원은 끕니다. 켜면 약 190px 간격으로 세로 줄무늬가 남습니다. L4(24GB)는 한 장 전체를 복원할 수 있습니다.
    return vae.eval().requires_grad_(False).to(DEVICE)


def load_transformer(folder: Path):
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel

    config = json.loads((folder / "transformer/config.json").read_text(encoding="utf-8"))
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(config)
        for block in model.transformer_blocks:
            old = block.img_mlp
            block.img_mlp = FusedMLP(old.gate_layer.in_features, old.gate_layer.out_features)
    path = model_file("qwen_transformer")
    header = read_safetensors_header(path)
    names = {key: key for key in header if key != ".comfy_attention.config"}
    load_weights_into_empty_model(model, path, check_names(model, header, names), DEVICE)
    return model.eval().requires_grad_(False)


def load_text_encoder(folder: Path):
    """문장 인코더(Qwen3-VL 8B)는 시스템 메모리에 두고, 쓸 때만 그래픽 카드로 옮깁니다."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig.from_pretrained(folder / "text_encoder", local_files_only=True)
    config._attn_implementation = config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)
    model.lm_head = nn.Identity()  # 단어 예측은 쓰지 않고 은닉 상태만 씁니다
    embed = model.model.language_model.embed_tokens
    model.model.language_model.embed_tokens = nn.Identity()
    path = model_file("qwen_text_encoder")
    header = read_safetensors_header(path)
    ignored = {key for key in header if key.startswith(("lm_head.", "model.embed_tokens."))}
    names = {key.replace("model.layers.", "model.language_model.layers.")
             .replace("model.norm.", "model.language_model.norm."): key
             for key in header if key not in ignored}
    check_names(model, {k: v for k, v in header.items() if k not in ignored}, names)
    load_weights_into_empty_model(model, path, names, "cpu")
    with safe_open(str(path), framework="pt", device="cpu") as source:
        prefix = "model.embed_tokens."
        tensors = {prefix + suffix: source.get_tensor(prefix + suffix) for suffix in ("comfy_quant", "weight_scale")}
        check_int8(tensors, prefix)
        weight = source.get_tensor(prefix + "weight")
        if tuple(weight.shape) != (embed.num_embeddings, embed.embedding_dim):
            raise ValueError("임베딩 모양이 설정과 다릅니다")
        model.model.language_model.embed_tokens = Int8Embedding(weight, layer_scale(tensors, prefix),
                                                                 convrot_group(tensors, prefix))
    return model.eval().requires_grad_(False)


class QwenImageModel:
    """한 번 불러와 여러 번 씁니다. edit()은 이미지 한 장, edit_with_reference()는 참고 이미지와 함께 편집합니다."""

    def __init__(self):
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
        from transformers import Qwen3VLProcessor

        folder = model_folder("qwen_configs")  # 파일이 없으면 여기서 받는 방법을 알려 주는 오류가 납니다
        self.text_encoder = load_text_encoder(folder)
        # 원본 예제의 simple 시그마를 그대로 넘기므로 diffusers가 한 번 더 이동시키지 않게 합니다
        scheduler = FlowMatchEulerDiscreteScheduler(shift=1, use_dynamic_shifting=False, shift_terminal=None)
        self.pipe = QwenImage21Pipeline(scheduler=scheduler, vae=load_vae(folder), text_encoder=None,
                                        processor=Qwen3VLProcessor.from_pretrained(folder / "processor",
                                                                                   local_files_only=True),
                                        transformer=load_transformer(folder))
        self.current_lora = None
        register_qwen21_blocks()
        # None이 아니면 MagCache 없이 모든 단계를 계산하며 크기 비율을 잽니다 (calibrate_mag_cache.py가 씀)
        self.calibration: dict[str, list[list[float]]] | None = None
        encode_prompt = self.pipe.encode_prompt

        def encode_on_gpu(*args, **kwargs):
            self.pipe.text_encoder = self.text_encoder.to(DEVICE)
            try:
                return encode_prompt(*args, **kwargs)
            finally:
                self.pipe.text_encoder = None
                self.text_encoder.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()

        self.pipe.encode_prompt = encode_on_gpu

    def use_lora(self, model_key: str | None, strength: float) -> None:
        """model_key의 LoRA를 붙입니다. None이면 LoRA 없이 기본 편집 모델로 씁니다."""
        if self.current_lora != (model_key, strength):
            if model_key is None:
                remove_loras(self.pipe.transformer)
            else:
                attach_lora(self.pipe.transformer, model_file(model_key), strength)
            self.current_lora = (model_key, strength)

    def run_steps(self, generate, steps: int):
        """generate()를 MagCache를 켠 채로 실행합니다. 크기 비율은 붙어 있는 LoRA마다 따로 씁니다."""
        name = (self.current_lora or (None,))[0] or "base"
        transformer = self.pipe.transformer
        if transformer.is_cache_enabled:
            transformer.disable_cache()
        if self.calibration is not None:
            recorder = StepChangeRecorder(transformer)
            try:
                return generate()
            finally:
                self.calibration.setdefault(name, []).append(recorder.remove())
        config = mag_cache_config(name, steps)
        if config is not None:
            transformer.enable_cache(config)
        return generate()

    @property
    def reuse_condition_cache(self) -> bool:
        """글자와 참고 이미지 부분은 단계마다 같으므로 첫 단계에서 계산해 두고 재사용합니다 (결과는 같음).
        크기 비율을 잴 때는 늘 켭니다: 둘째 단계부터 결과 토큰만 비교해야 비율이 맞습니다."""
        return self.calibration is not None or load_settings()["speed"]["reuse_condition_cache"]

    @torch.inference_mode()
    def edit(self, image: Image.Image, prompt: str, seed: int, steps: int) -> Image.Image:
        """이미지 한 장을 프롬프트대로 편집합니다. 결과 크기는 aligned_size(입력 크기)입니다.
        이 모델의 VAE는 투명도까지 4채널로 그리지만, 사진에는 투명도가 없으므로 RGB만 돌려줍니다
        (투명한 채로 크기를 바꾸면 색이 깨집니다)."""
        image = image.convert("RGB").resize(aligned_size(*image.size), Image.Resampling.LANCZOS)
        return self.run_steps(lambda: self.pipe(
            prompt=prompt, image=image, width=image.width, height=image.height,
            output_resolution=math.sqrt(image.width * image.height), true_cfg_scale=1, num_inference_steps=steps,
            sigmas=simple_sigmas(steps), use_kv_cache=self.reuse_condition_cache,
            generator=torch.Generator(device="cpu").manual_seed(seed)).images[0], steps).convert("RGB")

    @torch.inference_mode()
    def edit_with_reference(self, reference: Image.Image, image: Image.Image, prompt: str, seed: int,
                            steps: int) -> Image.Image:
        """reference(1번 이미지)를 참고해 image(2번 이미지)를 다시 그립니다. 결과 크기는 aligned_size(image 크기)입니다."""
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        pipe = self.pipe
        # 이 VAE는 RGBA 4채널을 받습니다 (pipe()는 내부에서 바꿔 주지만, 여기서는 직접 전처리하므로 맞춰 줍니다)
        images = [im.convert("RGBA").resize(aligned_size(*im.size), Image.Resampling.LANCZOS) for im in (reference, image)]
        width, height = images[-1].size
        text, text_mask, image_mask = pipe.encode_prompt(image=images, prompt=prompt, device=torch.device(DEVICE),
                                                         num_images_per_prompt=1)
        pixels = [pipe.image_processor.preprocess(im, height=im.height, width=im.width).unsqueeze(2) for im in images]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents, references = pipe.prepare_latents(pixels, 1, pipe.transformer.config.in_channels, height, width,
                                                    text.dtype, torch.device(DEVICE), generator)
        stride = pipe.vae_scale_factor
        shapes = [[(1, im.height // stride, im.width // stride) for im in [*images, images[-1]]]]
        image_mask = torch.cat([image_mask, image_mask.new_ones((1, latents.shape[1] // 4))], dim=1)
        pipe.scheduler.set_timesteps(sigmas=simple_sigmas(steps), device=DEVICE)
        pipe.scheduler.set_begin_index(0)
        cache = QwenImage21KVCache(len(pipe.transformer.transformer_blocks)) if self.reuse_condition_cache else None

        def denoise(latents):
            for index, t in enumerate(pipe.progress_bar(pipe.scheduler.timesteps)):
                mode = ("extract" if index == 0 else "cached") if cache else None
                prediction = pipe.transformer(hidden_states=torch.cat([references, latents], 1),
                                              timestep=t.expand(1).to(latents) / 1000, encoder_hidden_states=text,
                                              encoder_hidden_states_mask=text_mask, img_shapes=shapes,
                                              img_mask=image_mask, attention_kwargs={}, kv_cache=cache,
                                              kv_cache_mode=mode, return_dict=False)[0]
                latents = pipe.scheduler.step(prediction[:, -latents.size(1):], t, latents,
                                              return_dict=False)[0].to(latents.dtype)
            return latents

        latents = self.run_steps(lambda: denoise(latents), steps)
        latents = pipe._unpack_latents(latents, height, width, stride).to(pipe.vae.dtype)
        mean = latents.new_tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)
        decoded = pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0].convert("RGB")
