"""MagCache로 변화가 작은 생성 단계의 계산을 건너뜁니다.

생성은 여러 단계에 걸쳐 그림을 조금씩 고치는데, 뒤쪽 단계로 갈수록 한 단계의 변화량이 앞 단계와 거의 같아집니다.
MagCache는 단계마다 변화량이 앞 단계 대비 몇 배인지(크기 비율)를 미리 재 두고, 누적 오차가 작으면
그 단계의 블록 계산을 건너뛰고 앞 단계의 변화량을 다시 씁니다 (https://github.com/Zehong-Ma/MagCache).

크기 비율은 사진 한 장이 아니라 모델, LoRA, 단계 수에 따라 정해집니다. 기능마다 자기 사진 여러 장으로 한 번 재서
파일로 두고(save_ratios), 모델을 만들 때 그 파일을 넘깁니다 (QwenImageModel(ratios_file=...)).
"""

import json
import tomllib
from functools import cache
from pathlib import Path

import numpy as np
import torch

SETTINGS_FILE = Path(__file__).resolve().parent / "cache_settings.toml"


@cache
def cache_settings() -> dict:
    with open(SETTINGS_FILE, "rb") as file:
        return tomllib.load(file)


def register_qwen21_blocks() -> None:
    """diffusers의 MagCache가 Qwen-Image 2.1 블록의 입력과 출력 위치를 알도록 등록합니다 (0.43 기준 미등록)."""
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

    TransformerBlockRegistry.register(model_class=QwenImage21TransformerBlock,
                                      metadata=TransformerBlockMetadata(return_hidden_states_index=0,
                                                                        return_encoder_hidden_states_index=None))


def load_ratios(ratios_file: Path | None) -> dict[str, list[float]]:
    """{LoRA 이름 또는 base: 단계별 크기 비율}. 파일이 없으면 빈 사전 (MagCache 없이 모든 단계 계산)."""
    if ratios_file is None or not ratios_file.is_file():
        return {}
    return json.loads(ratios_file.read_text(encoding="utf-8"))["ratios"]


def save_ratios(ratios_file: Path, measured: dict[str, list[list[float]]], steps: int, photos: int) -> None:
    """잰 비율(이름마다 여러 번 잰 목록)을 평균 내서 저장합니다."""
    ratios = {name: [round(float(v), 5) for v in np.mean(runs, axis=0)] for name, runs in measured.items()}
    record = {"steps": steps, "photos": photos, "runs": {k: len(v) for k, v in measured.items()}, "ratios": ratios}
    ratios_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def mag_cache_config(ratios: list[float] | None, steps: int):
    """MagCache 설정. 꺼져 있거나 잰 비율이 없으면 None (모든 단계 계산)."""
    settings = cache_settings()
    if not settings["mag_cache"] or ratios is None:
        return None
    from diffusers import MagCacheConfig

    return MagCacheConfig(threshold=settings["mag_cache_threshold"], max_skip_steps=settings["mag_cache_max_skip_steps"],
                          retention_ratio=settings["mag_cache_retention_ratio"], num_inference_steps=steps,
                          mag_ratios=ratios)


def run_steps(transformer, generate, steps: int, ratios: list[float] | None, record: list | None = None):
    """generate()를 MagCache를 켠 채로 실행합니다.
    record가 리스트면 MagCache 없이 모든 단계를 계산하며, 이번 생성의 단계별 크기 비율을 그 리스트에 붙입니다."""
    if transformer.is_cache_enabled:
        transformer.disable_cache()
    if record is not None:
        recorder = StepChangeRecorder(transformer)
        try:
            return generate()
        finally:
            record.append(recorder.remove())
    config = mag_cache_config(ratios, steps)
    if config is not None:
        transformer.enable_cache(config)
    return generate()


class StepChangeRecorder:
    """크기 비율을 잽니다: 단계마다 (마지막 블록 출력 - 첫 블록 입력)의 크기가 앞 단계의 몇 배인지 기록합니다.

    조건 캐시를 켜면 둘째 단계부터는 결과 이미지 토큰만 블록을 지나므로, 두 단계 모두에 있는 맨 뒤 토큰끼리 비교합니다.
    """

    def __init__(self, transformer):
        blocks = transformer.transformer_blocks
        self.ratios, self.previous, self.block_input = [], None, None
        self.handles = [blocks[0].register_forward_pre_hook(self.keep_input, with_kwargs=True),
                        blocks[-1].register_forward_hook(self.measure)]

    def keep_input(self, module, args, kwargs):
        self.block_input = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

    def measure(self, module, args, output):
        change = (output - self.block_input).float()
        if self.previous is None:
            self.ratios.append(1.0)
        else:
            tokens = min(change.shape[1], self.previous.shape[1])
            now, before = (torch.linalg.norm(x[:, -tokens:], dim=-1) for x in (change, self.previous))
            self.ratios.append((now / (before + 1e-8)).mean().item())
        self.previous = change

    def remove(self) -> list[float]:
        for handle in self.handles:
            handle.remove()
        return self.ratios
