"""IC-Light(전경 조건 모델)로 상품에 스튜디오 조명을 새로 입히고 배경을 새로 그립니다.

공식 데모(lllyasviel/IC-Light gradio_demo.py)와 같은 방식입니다.
- 기반 SD1.5의 첫 합성곱을 8채널로 늘리고, IC-Light 가중치를 기존 가중치에 더함
- 상품을 회색 배경에 놓고 잠재 공간으로 바꿔 매 단계 함께 넣음
- 낮은 해상도에서 만들고, 키운 뒤 한 번 더 다듬음 (highres)
생성 뒤 상품 영역은 원본 색과 세부를 두고 새 조명의 밝기만 입힙니다(keep_product_color). 끄면 생성된 색에 원본 세부만 다시 입힙니다.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .adjustments import linear_to_srgb, luminance, srgb_to_linear
from .cleanup import resize
from .download_models import model_folder

DIRECTIONS = {"left": (1, 0), "right": (-1, 0), "top": (0, 1), "bottom": (0, -1)}


def light_gradient(direction: str, width: int, height: int) -> np.ndarray | None:
    """공식 데모의 빛 방향 배경: 밝은 쪽에서 어두운 쪽으로 가는 회색 기울기."""
    if direction not in DIRECTIONS:
        return None
    dx, dy = DIRECTIONS[direction]
    ramp = np.linspace(1, 0, width if dx else height, dtype=np.float32)
    ramp = ramp if (dx or dy) > 0 else ramp[::-1]
    plane = np.tile(ramp, (height, 1)) if dx else np.tile(ramp[:, None], (1, width))
    return np.repeat(plane[..., None], 3, axis=-1)


def size_for(width: int, height: int, long_side: float) -> tuple[int, int]:
    scale = long_side / max(width, height)
    return max(64, round(width * scale / 64) * 64), max(64, round(height * scale / 64) * 64)


class Relighter:
    def __init__(self):
        import torch
        from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline,
                               StableDiffusionPipeline, UNet2DConditionModel)
        from safetensors.torch import load_file
        from transformers import CLIPTextModel, CLIPTokenizer

        self.torch = torch
        base = model_folder("relight_base")
        dtype = torch.float16
        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
        with torch.no_grad():
            conv = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
            conv.weight.zero_()
            conv.weight[:, :4].copy_(unet.conv_in.weight)
            conv.bias = unet.conv_in.bias
            unet.conv_in = conv
        offset = load_file(model_folder("relight") / "iclight_sd15_fc.safetensors")
        unet.load_state_dict({k: v + offset[k] for k, v in unet.state_dict().items()}, strict=True)
        original_forward = unet.forward

        def forward(sample, timestep, encoder_hidden_states, **kwargs):
            extra = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
            extra = torch.cat([extra] * (sample.shape[0] // extra.shape[0]), dim=0)
            kwargs["cross_attention_kwargs"] = {}
            return original_forward(torch.cat([sample, extra], dim=1), timestep, encoder_hidden_states, **kwargs)

        unet.forward = forward
        parts = {
            "vae": AutoencoderKL.from_pretrained(base, subfolder="vae").to("cuda", dtype),
            "text_encoder": CLIPTextModel.from_pretrained(base, subfolder="text_encoder").to("cuda", dtype),
            "tokenizer": CLIPTokenizer.from_pretrained(base, subfolder="tokenizer"),
            "unet": unet.to("cuda", dtype),
            "scheduler": DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                                                     algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, steps_offset=1),
            "safety_checker": None, "feature_extractor": None, "requires_safety_checker": False,
        }
        self.vae = parts["vae"]
        self.text_to_image = StableDiffusionPipeline(**parts, image_encoder=None)
        self.image_to_image = StableDiffusionImg2ImgPipeline(**parts, image_encoder=None)

    def encode(self, image: np.ndarray):
        x = self.torch.from_numpy(image * 2 - 1).permute(2, 0, 1)[None].to("cuda", self.vae.dtype)
        return self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

    def decode(self, latents) -> np.ndarray:
        pixels = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor).sample
        return ((pixels[0].permute(1, 2, 0).float().cpu().numpy() + 1) / 2).clip(0, 1)

    def run(self, foreground: np.ndarray, prompt: str, rule: dict) -> np.ndarray:
        torch = self.torch
        height, width = foreground.shape[:2]
        common = {"prompt": f"{prompt}, {rule['added_prompt']}", "negative_prompt": rule["negative_prompt"],
                  "guidance_scale": rule["cfg"], "output_type": "latent",
                  "generator": torch.Generator("cuda").manual_seed(rule["seed"])}
        with torch.no_grad():
            size = size_for(width, height, rule["long_side"])
            condition = {"concat_conds": self.encode(resize(foreground, size))}
            gradient = light_gradient(rule["light_direction"], *size)
            if gradient is None:
                latents = self.text_to_image(width=size[0], height=size[1], num_inference_steps=rule["steps"],
                                             cross_attention_kwargs=condition, **common).images
            else:
                denoise = rule["lowres_denoise"]
                latents = self.image_to_image(image=self.encode(gradient), strength=denoise,
                                              num_inference_steps=round(rule["steps"] / denoise),
                                              cross_attention_kwargs=condition, **common).images
            size = size_for(width, height, rule["long_side"] * rule["highres_scale"])
            first = resize(self.decode(latents), size)
            denoise = rule["highres_denoise"]
            latents = self.image_to_image(image=self.encode(first), strength=denoise,
                                          num_inference_steps=round(rule["steps"] / denoise),
                                          cross_attention_kwargs={"concat_conds": self.encode(resize(foreground, size))},
                                          **common).images
            return resize(self.decode(latents), (width, height))

    def close(self):
        del self.text_to_image, self.image_to_image, self.vae
        self.torch.cuda.empty_cache()


def relight(image: np.ndarray, subject: np.ndarray, prompt: str, rule: dict) -> np.ndarray:
    weight = subject[..., None]
    foreground = image * weight + 0.5 * (1 - weight)  # 공식 데모처럼 상품 밖은 회색
    relighter = Relighter()
    lit = relighter.run(foreground, prompt, rule)
    relighter.close()
    sigma = rule["detail_sigma_px"]
    if rule["keep_product_color"]:
        # 상품은 원본 색과 세부를 그대로 두고, 새 조명의 밝기 분포(그림자, 하이라이트)만 곱함
        # 상품 안쪽만 넣어 흐리게 하므로 테두리에서 배경 밝기가 섞이지 않음 (가중치는 나눗셈에서 지워짐)
        before, after = srgb_to_linear(image), srgb_to_linear(lit)
        spread, inside = rule["shade_sigma"] * max(image.shape[:2]), subject
        shade = (gaussian_filter(luminance(after) * inside, spread)
                 / np.maximum(gaussian_filter(luminance(before) * inside, spread), 1e-4))
        relit = before * np.clip(shade, *rule["shade_range"])[..., None]
        # 한 채널이 끝을 넘으면 세 채널을 같은 비율로 낮춰 색이 하얗게 바래지 않게 함
        product = linear_to_srgb(relit / np.maximum(relit.max(axis=-1, keepdims=True), 1.0))
    else:
        detail = image - np.stack([gaussian_filter(image[..., c], sigma) for c in range(3)], axis=-1)
        product = np.stack([gaussian_filter(lit[..., c], sigma) for c in range(3)], axis=-1) + detail
    # 반투명 테두리에는 원래 배경색이 섞여 있어 띠로 남으므로, 그 띠는 새 조명 쪽 화소를 씀
    low, high = rule["edge_band"]
    solid = np.clip((subject - low) / (high - low), 0, 1)[..., None]
    return np.clip(lit * (1 - solid) + product * solid, 0, 1).astype(np.float32)
