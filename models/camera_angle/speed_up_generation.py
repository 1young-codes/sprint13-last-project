"""생성 속도 올리기: MagCache로 변화가 작은 단계의 계산을 건너뜁니다.

생성은 25단계에 걸쳐 그림을 조금씩 고치는데, 뒤쪽 단계로 갈수록 한 단계의 변화량이 앞 단계와 거의 같아집니다.
MagCache는 단계마다 변화량이 앞 단계 대비 몇 배인지(크기 비율)를 미리 재 두고, 누적 오차가 작으면
그 단계의 블록 계산을 건너뛰고 앞 단계의 변화량을 다시 씁니다 (https://github.com/Zehong-Ma/MagCache).

크기 비율은 사진이 아니라 모델과 LoRA에 따라 정해지는 값이라 mag_cache_ratios.json에 저장해 둡니다.
모델, LoRA, 단계 수를 바꾸면 다시 잽니다:
    python -m models.camera_angle.calibrate_mag_cache --input 사진1.jpg --input 사진2.jpg --preset all
"""

import json

import torch

from .settings import PACKAGE_DIR, load_settings

RATIOS_FILE = PACKAGE_DIR / "mag_cache_ratios.json"


def register_qwen21_blocks() -> None:
    """diffusers의 MagCache가 Qwen-Image 2.1 블록의 입력과 출력 위치를 알도록 등록합니다 (0.43 기준 미등록)."""
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

    TransformerBlockRegistry.register(model_class=QwenImage21TransformerBlock,
                                      metadata=TransformerBlockMetadata(return_hidden_states_index=0,
                                                                        return_encoder_hidden_states_index=None))


def load_ratios() -> dict[str, list[float]]:
    return json.loads(RATIOS_FILE.read_text(encoding="utf-8"))["ratios"] if RATIOS_FILE.is_file() else {}


def mag_cache_config(name: str, steps: int):
    """name(LoRA 이름 또는 base)에 맞는 MagCache 설정. 꺼져 있거나 잰 비율이 없으면 None (모든 단계 계산)."""
    from diffusers import MagCacheConfig

    settings = load_settings()["speed"]
    ratios = load_ratios().get(name)
    if not settings["mag_cache"] or ratios is None:
        return None
    return MagCacheConfig(threshold=settings["mag_cache_threshold"], max_skip_steps=settings["mag_cache_max_skip_steps"],
                          retention_ratio=settings["mag_cache_retention_ratio"], num_inference_steps=steps,
                          mag_ratios=ratios)


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
