"""캐싱 계산 검증 (그래픽 카드와 모델 없이 작은 가짜 모델로 확인)."""

import pytest
import torch
from torch import nn

from models.generative_ai.fast_image_generation.qwen_image_caching.mag_cache import (
    StepChangeRecorder, load_ratios, mag_cache_config, run_steps, save_ratios)


class Shift(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1


class TinyTransformer(nn.Module):
    """블록 두 개짜리 가짜 생성 모델. MagCache를 켜려 하면 실패합니다 (비율을 잴 때는 켜면 안 되므로)."""
    is_cache_enabled = False

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Identity(), Shift()])

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x

    def enable_cache(self, config):
        raise AssertionError("비율을 잴 때는 MagCache를 켜면 안 됩니다")


def scaled_steps(transformer):
    """두 단계: 둘째 단계의 변화량은 같고, 조건 캐시처럼 앞쪽 토큰 2개가 빠집니다."""
    transformer(torch.zeros(1, 6, 3))
    transformer(torch.zeros(1, 4, 3))


def test_recorder_compares_only_the_tokens_both_steps_share():
    transformer = TinyTransformer()
    recorder = StepChangeRecorder(transformer)
    scaled_steps(transformer)
    assert recorder.remove() == pytest.approx([1.0, 1.0])


def test_measuring_runs_every_step_without_mag_cache():
    transformer, record = TinyTransformer(), []
    run_steps(transformer, lambda: scaled_steps(transformer), steps=2, ratios=[1.0, 1.0], record=record)
    assert record == [pytest.approx([1.0, 1.0])]


def test_without_measured_ratios_every_step_is_computed():
    assert mag_cache_config(None, 25) is None


def test_saved_ratios_are_averaged_per_lora(tmp_path):
    path = tmp_path / "ratios.json"
    save_ratios(path, {"base": [[1.0, 2.0], [3.0, 4.0]]}, steps=2, photos=1)
    assert load_ratios(path) == {"base": [2.0, 3.0]}
    assert load_ratios(tmp_path / "없는파일.json") == {}
