"""MOSS-Image SFT로 글만 보고 이미지를 만듭니다 (그래픽 카드 필요, ComfyUI 없이 공식 파이프라인 사용).

    python -m models.generative_ai.fast_image_generation.moss_image_model    # 모델 약 27GB를 HF_HOME 캐시에 한 번만 받음

diffusers 본체에는 MOSS가 없어서, OpenMOSS가 모델 저장소에 올린 diffusers용 파이프라인 파일(pipeline_moss_image.py)을
고정 커밋으로 받아 씁니다. 가중치는 공식 bf16 그대로입니다. 로라 배포 페이지에 정수 8비트 가중치에서는 로라 효과가
약해진다고 적혀 있어서, Qwen 쪽처럼 INT8 파일을 쓰지 않습니다.

L4(24GB)와 시스템 메모리 15GB에 맞춘 두 단계 실행:
1. encode_prompts(): 문장 인코더(Qwen3.5-9B, 약 17GB)를 그래픽 카드에 바로 올려 프롬프트를 모두 인코딩하고 파일로 둔 뒤 내립니다.
   같은 프롬프트는 다시 인코딩하지 않습니다 (결과가 같은 조건 캐시).
2. MossImageModel: 생성 모델(약 10GB)을 올리고 로라를 가중치에 합친 뒤, 저장해 둔 인코딩으로 생성합니다.
   MagCache로 변화가 작은 단계는 건너뜁니다.

Qwen과 달리 MOSS는 글 토큰도 매 단계 그림 토큰과 같은 블록을 지나며 단계마다 다른 조정값을 받습니다.
그래서 블록 안의 글 부분 계산을 첫 단계에서 저장해 다시 쓰는 방식은 결과를 바꾸므로 쓰지 않습니다.
"""

import gc
import hashlib
import importlib.util
import itertools
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from .model_files import download_model, model_folder, read_sources
from .qwen_image_caching.mag_cache import load_ratios, mag_cache_config

DEVICE = "cuda"
MODEL_KEY = "moss_sft"
SOURCES_FILE = Path(__file__).resolve().parent / "moss_model_sources.toml"


@cache
def official_code():
    """모델 저장소의 pipeline_moss_image.py (MOSSImagePipeline, MOSSImageTransformer, MOSSImageScheduler)."""
    path = model_folder(MODEL_KEY, SOURCES_FILE) / "pipeline_moss_image.py"
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def embedding_file(cache_dir: Path, prompt: str) -> Path:
    return cache_dir / f"{hashlib.sha256(prompt.encode()).hexdigest()[:20]}.pt"


def encode_prompts(prompts: list[str], cache_dir: Path) -> dict[str, Path]:
    """프롬프트마다 인코딩 파일 경로. 이미 있는 것은 건너뛰고, 없는 것만 인코더를 올려 만듭니다."""
    from transformers import AutoModel, AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {prompt: embedding_file(cache_dir, prompt) for prompt in dict.fromkeys(prompts)}
    missing = [prompt for prompt, file in files.items() if not file.is_file()]
    if not missing:
        return files
    folder = model_folder(MODEL_KEY, SOURCES_FILE)
    tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_pretrained(folder / "text_encoder", dtype=torch.bfloat16, device_map=DEVICE,
                                        local_files_only=True).eval()
    # 공식 encode_prompt를 그대로 부릅니다 (채팅 형식, thinking 켬, 마지막 은닉 상태)
    owner = SimpleNamespace(text_encoder=encoder, tokenizer=tokenizer, _execution_device=torch.device(DEVICE))
    try:
        for prompt in missing:
            embeds, _ = official_code().MOSSImagePipeline.encode_prompt(owner, prompt)
            torch.save(embeds[0].cpu(), files[prompt])
    finally:
        del owner, encoder
        gc.collect()
        torch.cuda.empty_cache()
    return files


def lora_deltas(weights: dict[str, torch.Tensor], strength: float) -> dict[str, torch.Tensor]:
    """로라 파일 → {층 이름: 가중치에 더할 값}. 두 형식을 받습니다.

    - LoRA(PEFT): <층>.lora_A.weight, <층>.lora_B.weight, (선택) <층>.alpha. 더할 값 = B·A × alpha/rank
    - LoKr(완전 행렬): <층>.lokr_w1, <층>.lokr_w2. 더할 값 = w1 ⊗ w2. 두 행렬이 분해되지 않았으므로 alpha는 쓰지 않습니다
      (ComfyUI weight_adapter/lokr.py와 같음). 이 파일의 q, k, v 자리 alpha 키는 짝이 없어 그냥 넘깁니다.
    proj.qkv는 이 모델에서 q, k, v 세 층이라 행을 셋으로 나눕니다 ([q; k; v] 순서).
    """
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in weights.items():
        suffix = next((s for s in (".lora_A.weight", ".lora_B.weight", ".lokr_w1", ".lokr_w2", ".alpha")
                       if key.endswith(s)), None)
        if suffix is None:
            raise ValueError(f"지원하지 않는 로라 키입니다: {key}")
        groups.setdefault(key[:-len(suffix)].removeprefix("diffusion_model."), {})[suffix] = value.float()
    deltas = {}
    for name, parts in groups.items():
        if ".lora_A.weight" in parts:
            down, up = parts[".lora_A.weight"], parts[".lora_B.weight"]
            alpha = parts.get(".alpha")
            delta = up @ down * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
        elif ".lokr_w1" in parts:
            delta = torch.kron(parts[".lokr_w1"], parts[".lokr_w2"])
        else:
            continue  # alpha만 있는 자리
        delta = delta * strength
        if name.endswith(".qkv"):
            for part, rows in zip("qkv", delta.chunk(3, dim=0)):
                deltas[f"{name[:-3]}{part}"] = rows
        else:
            deltas[name] = delta
    return deltas


def merge_lora(transformer: torch.nn.Module, lora_file: Path, strength: float) -> int:
    """로라를 가중치에 합칩니다. 합친 층 수를 돌려줍니다. 떼려면 모델을 다시 불러옵니다."""
    from safetensors.torch import load_file

    deltas = lora_deltas(load_file(str(lora_file)), strength)
    for name, delta in deltas.items():
        layer = transformer.get_submodule(name)
        if not isinstance(layer, torch.nn.Linear) or tuple(layer.weight.shape) != tuple(delta.shape):
            raise ValueError(f"{lora_file.name}: {name}의 모양이 모델과 다릅니다")
        weight = layer.weight.data
        weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
    return len(deltas)


def register_moss_blocks() -> None:
    """diffusers MagCache가 MOSS 블록의 입력(x)과 출력 위치를 알도록 등록합니다."""
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry

    TransformerBlockRegistry.register(model_class=official_code()._Block,
                                      metadata=TransformerBlockMetadata(return_hidden_states_index=0,
                                                                        hidden_states_argument_name="x"))


class ContextRecorder:
    """MagCache 크기 비율을 잽니다. CFG라 한 단계에 긍정·부정 두 번 돌므로, 갈래마다 따로 앞 단계와 비교합니다."""

    def __init__(self, blocks):
        self.context, self.block_input, self.previous, self.ratios = "cond", None, {}, {}
        self.handles = [blocks[0].register_forward_pre_hook(self.keep_input),
                        blocks[-1].register_forward_hook(self.measure)]

    def keep_input(self, module, args):
        self.block_input = args[0]

    def measure(self, module, args, output):
        change = (output - self.block_input).float()
        before = self.previous.get(self.context)
        ratio = 1.0 if before is None else (torch.linalg.norm(change, dim=-1) /
                                            (torch.linalg.norm(before, dim=-1) + 1e-8)).mean().item()
        self.ratios.setdefault(self.context, []).append(ratio)
        self.previous[self.context] = change

    def remove(self) -> dict[str, list[float]]:
        for handle in self.handles:
            handle.remove()
        return self.ratios


class MossImageModel:
    """조건(로라) 하나로 불러와 여러 장을 만듭니다. 로라를 바꾸려면 새로 만듭니다 (로라를 가중치에 합치므로)."""

    def __init__(self, lora_file: Path | None = None, strength: float = 1.0, ratios_file: Path | None = None):
        """ratios_file: MagCache 크기 비율 파일 ({조건 이름: 비율}). 없거나 이 조건이 없으면 모든 단계를 계산합니다."""
        from diffusers import AutoencoderKLFlux2

        code, folder = official_code(), model_folder(MODEL_KEY, SOURCES_FILE)
        transformer = code.MOSSImageTransformer.from_pretrained(folder / "transformer", torch_dtype=torch.bfloat16,
                                                                device_map=DEVICE, local_files_only=True).eval()
        vae = AutoencoderKLFlux2.from_pretrained(folder / "vae", torch_dtype=torch.float32,
                                                 local_files_only=True).eval().to(DEVICE)
        scheduler = code.MOSSImageScheduler.from_pretrained(folder / "scheduler", local_files_only=True)
        self.pipe = code.MOSSImagePipeline(transformer=transformer, scheduler=scheduler, vae=vae)
        self.pipe.set_progress_bar_config(disable=True)
        self.name = lora_file.stem if lora_file else "base"
        self.merged_layers = merge_lora(transformer, lora_file, strength) if lora_file else 0
        register_moss_blocks()
        self.ratios = load_ratios(ratios_file).get(self.name)
        # 리스트를 넣으면 MagCache 없이 모든 단계를 계산하며, 생성마다 긍정 갈래의 단계별 크기 비율을 붙입니다
        self.calibration: list[list[float]] | None = None

    @torch.inference_mode()
    def generate(self, embeds: torch.Tensor, negative: torch.Tensor, width: int, height: int, seed: int,
                 steps: int, cfg: float) -> Image.Image:
        """embeds, negative: encode_prompts()로 저장한 인코딩 [토큰, 4096]."""
        from diffusers.hooks import HookRegistry
        from diffusers.hooks.mag_cache import _MAG_CACHE_BLOCK_HOOK, _MAG_CACHE_LEADER_BLOCK_HOOK, apply_mag_cache

        transformer = self.pipe.transformer
        config = None if self.calibration is not None else mag_cache_config(self.ratios, steps)
        recorder = ContextRecorder(transformer.layers) if self.calibration is not None else None
        registry = HookRegistry.check_if_exists_or_initialize(transformer)
        if config is not None:
            apply_mag_cache(transformer, config)
        # 공식 반복문은 한 단계에 긍정, 부정 순서로 생성 모델을 부릅니다. 갈래마다 MagCache 상태를 따로 둡니다
        contexts = itertools.cycle(["cond", "uncond"] if cfg != 1 else ["cond"])

        def forward(*args, **kwargs):
            name = next(contexts)
            registry._set_context(name)
            if recorder:
                recorder.context = name
            return type(transformer).forward(transformer, *args, **kwargs)

        transformer.forward = forward
        try:
            image = self.pipe(prompt_embeds=embeds[None].to(DEVICE), negative_prompt_embeds=negative[None].to(DEVICE),
                              width=width, height=height, num_inference_steps=steps, guidance_scale=cfg,
                              generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
        finally:
            del transformer.forward
            registry._set_context(None)
            if config is not None:
                registry.remove_hook(_MAG_CACHE_LEADER_BLOCK_HOOK, recurse=True)
                registry.remove_hook(_MAG_CACHE_BLOCK_HOOK, recurse=True)
            if recorder:
                self.calibration.append(recorder.remove()["cond"])
        return image


if __name__ == "__main__":
    for key, entry in read_sources(SOURCES_FILE)["models"].items():
        download_model(key, entry)
