"""그래픽 카드 없이 MOSS 로라 합치기와 MagCache 비율 재기를 검증합니다."""

import torch
from safetensors.torch import save_file
from torch import nn

from models.generative_ai.fast_image_generation.moss_image_model import ContextRecorder, lora_deltas, merge_lora


def tiny_model(dim=8):
    block = nn.Module()
    block.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in "qkvo"})
    model = nn.Module()
    model.layers = nn.ModuleList([block])
    return model


def test_peft_lora_is_up_times_down_with_alpha():
    down, up = torch.randn(2, 8), torch.randn(8, 2)
    weights = {"diffusion_model.layers.0.proj.o.lora_A.weight": down,
               "diffusion_model.layers.0.proj.o.lora_B.weight": up,
               "diffusion_model.layers.0.proj.o.alpha": torch.tensor(1.0)}
    delta = lora_deltas(weights, strength=0.5)["layers.0.proj.o"]
    torch.testing.assert_close(delta, up @ down * (1.0 / 2) * 0.5)


def test_lokr_fused_qkv_splits_into_q_k_v_and_ignores_alpha():
    w1, w2 = torch.randn(3, 1), torch.randn(8, 8)  # kron → (24, 8) = [q; k; v]
    weights = {"diffusion_model.layers.0.proj.qkv.lokr_w1": w1, "diffusion_model.layers.0.proj.qkv.lokr_w2": w2,
               **{f"diffusion_model.layers.0.proj.{p}.alpha": torch.tensor(1e10) for p in "qkv"}}
    deltas = lora_deltas(weights, strength=1.0)
    full = torch.kron(w1, w2)
    assert set(deltas) == {"layers.0.proj.q", "layers.0.proj.k", "layers.0.proj.v"}
    torch.testing.assert_close(deltas["layers.0.proj.k"], full[8:16])


def test_merge_adds_delta_and_refuses_wrong_shape(tmp_path):
    model = tiny_model()
    before = model.layers[0].proj["o"].weight.clone()
    down, up = torch.randn(2, 8), torch.randn(8, 2)
    good = tmp_path / "good.safetensors"
    save_file({"diffusion_model.layers.0.proj.o.lora_A.weight": down,
               "diffusion_model.layers.0.proj.o.lora_B.weight": up}, str(good))
    assert merge_lora(model, good, 1.0) == 1
    torch.testing.assert_close(model.layers[0].proj["o"].weight, before + up @ down)

    bad = tmp_path / "bad.safetensors"
    save_file({"diffusion_model.layers.0.proj.o.lora_A.weight": torch.randn(2, 4),
               "diffusion_model.layers.0.proj.o.lora_B.weight": torch.randn(8, 2)}, str(bad))
    try:
        merge_lora(model, bad, 1.0)
    except ValueError as error:
        assert "모양" in str(error)
    else:
        raise AssertionError("모양이 다른 로라를 받아들였습니다")


def test_recorder_compares_each_branch_with_its_own_previous_step():
    blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    blocks[1].register_forward_hook(lambda m, a, out: out * 2)  # 변화량 = 입력
    recorder = ContextRecorder(blocks)
    for step_value in (1.0, 2.0):           # 긍정 변화량 크기가 1 → 2: 비율 2
        for context, scale in (("cond", 1.0), ("uncond", 10.0)):
            recorder.context = context
            blocks[1](blocks[0](torch.full((1, 3, 4), step_value * scale)))
    ratios = recorder.remove()
    assert ratios["cond"] == [1.0, 2.0]
    assert ratios["uncond"] == [1.0, 2.0]
