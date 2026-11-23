"""정수 8비트(INT8)로 저장된 선형 층과, 그 가중치를 빈 모델에 채우는 함수.

Comfy-Org가 배포한 Qwen-Image 2.1 파일은 가중치를 INT8 + ConvRot 회전으로 저장합니다.
파일 값 q = round(W·H / s), H는 그룹마다 적용하는 정규화 아다마르 행렬, s는 행마다 하나인 배율입니다.
ai-generation 프로젝트의 backend/studio/engine/quant.py와 wan.py의 fill에서 INT8 부분만 옮겼습니다.
"""

import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

try:  # comfy-kitchen: ComfyUI의 INT8 ConvRot 전용 커널. 없으면 torch 연산으로 계산합니다(더 느림)
    import comfy_kitchen
except ImportError:
    comfy_kitchen = None

INT8 = "int8_tensorwise"
QUANT_SUFFIXES = (".comfy_quant", ".weight_scale")  # 가중치가 아니라 양자화 정보를 담은 키
ROWS = 4096  # 한 번에 처리하는 행 수 (실수 중간값이 층 전체 크기로 커지지 않게)


def read_safetensors_header(path: Path) -> dict:
    """텐서를 읽지 않고 safetensors 파일의 이름표와 모양만 읽습니다."""
    with open(path, "rb") as file:
        (size,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(size))
    header.pop("__metadata__", None)
    return header


def quant_config(tensors: dict, prefix: str) -> dict | None:
    key = f"{prefix}comfy_quant"
    return json.loads(bytes(tensors[key].tolist()).decode()) if key in tensors else None


def check_int8(tensors: dict, prefix: str) -> None:
    config = quant_config(tensors, prefix)
    if config is None or config.get("format") != INT8:
        raise ValueError(f"{prefix.rstrip('.')}: INT8 형식이 아닙니다 ({config})")


def convrot_group(tensors: dict, prefix: str) -> int:
    """ConvRot 회전 그룹 크기. 회전하지 않은 층은 0."""
    config = quant_config(tensors, prefix) or {}
    return int(config.get("convrot_groupsize", 256)) if config.get("convrot") else 0


def layer_scale(tensors: dict, prefix: str) -> torch.Tensor:
    return tensors[f"{prefix}weight_scale"].to(torch.float32, copy=True).reshape(-1, 1)


_HADAMARD: dict = {}


def hadamard(size: int, device=None) -> torch.Tensor:
    """ConvRot이 쓰는 정규화 아다마르 행렬 (4×4 정규 행렬의 크로네커 거듭제곱 / √size)."""
    key = (size, str(device))
    if key not in _HADAMARD:
        if size < 4 or size & (size - 1) or (size.bit_length() - 1) % 2:
            raise ValueError(f"ConvRot 그룹은 4의 거듭제곱이어야 합니다: {size}")
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = h4
        while h.shape[0] < size:
            h = torch.kron(h, h4)
        _HADAMARD[key] = (h / size ** 0.5).to(device)
    return _HADAMARD[key]


def rotate(x: torch.Tensor, group: int) -> torch.Tensor:
    """마지막 차원을 group개씩 묶어 x·H를 계산합니다."""
    shape = x.shape
    return (x.reshape(-1, shape[-1] // group, group) @ hadamard(group, x.device).to(x.dtype)).reshape(shape)


class Int8Linear(nn.Module):
    """가중치가 INT8인 선형 층. loras에는 실행 중에 더하는 LoRA 항 (A, B, 배율)이 들어갑니다:
    y += 배율 · (x Aᵀ) Bᵀ. 저장된 가중치는 바꾸지 않으므로 LoRA를 바꿔 끼우기 쉽습니다."""

    def __init__(self, in_features: int, out_features: int, bias: bool, convrot: int):
        super().__init__()
        if convrot and in_features % convrot:
            raise ValueError(f"ConvRot 그룹 {convrot}이 입력 크기 {in_features}를 나누지 못합니다")
        self.in_features, self.out_features, self.convrot = in_features, out_features, convrot
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None
        self.register_buffer("scale", torch.ones(out_features, 1), persistent=False)
        self.loras: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        if comfy_kitchen is not None and x.is_cuda and self.convrot in (0, 256):
            y = comfy_kitchen.int8_linear(x, self.weight, self.scale, bias, x.dtype, convrot=bool(self.convrot),
                                          convrot_groupsize=self.convrot or 256, input_act=None)
        else:
            y = self.int8_matmul(x, bias)
        for a, b, s in self.loras:
            a, b = a.to(x.device, x.dtype), b.to(x.device, x.dtype)
            y.view(-1, self.out_features).addmm_(x.reshape(-1, self.in_features) @ a.T, b.T, alpha=s)
        return y

    def int8_matmul(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        """입력도 행마다 INT8로 바꿔 정수 곱셈을 하고, 두 배율을 다시 곱합니다. x Wᵀ = (x H)(q · s)ᵀ."""
        x2 = x.reshape(-1, self.in_features)
        out = torch.empty(x2.shape[0], self.out_features, dtype=x.dtype, device=x.device)
        qt, s = self.weight.t(), self.scale.view(1, -1)
        for i in range(0, x2.shape[0], ROWS):
            xs = rotate(x2[i:i + ROWS], self.convrot) if self.convrot else x2[i:i + ROWS]
            xscale = (xs.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
            xq = (xs.float() / xscale).round_().clamp_(-128, 127).to(torch.int8)
            n = xq.shape[0]
            if n <= 16:  # cuBLASLt INT8 곱셈은 16행보다 많아야 합니다
                xq = torch.cat([xq, xq.new_zeros(17 - n, xq.shape[1])])
            out[i:i + n] = torch._int_mm(xq, qt)[:n].float().mul_(xscale).mul_(s)
        if bias is not None:
            out += bias
        return out.reshape(*x.shape[:-1], self.out_features)


def replace_linears_with_int8(model: nn.Module, groups: dict[str, int]) -> list[str]:
    """groups(층 이름 + '.' → ConvRot 그룹)에 있는 nn.Linear를 같은 모양의 Int8Linear로 바꿉니다."""
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and f"{name}." in groups:
            parent, _, child = name.rpartition(".")
            with torch.device("meta"):
                layer = Int8Linear(module.in_features, module.out_features, module.bias is not None, groups[f"{name}."])
            setattr(model.get_submodule(parent) if parent else model, child, layer)
            replaced.append(name)
    return replaced


def load_weights_into_empty_model(model: nn.Module, path: Path, names: dict[str, str], device: str,
                                  dtype=torch.bfloat16) -> None:
    """빈(meta) 모델을 safetensors 파일로 채웁니다. names: 모델 키 → 파일 키.
    INT8로 저장된 층은 Int8Linear로 바꾸고, 나머지 실수 텐서는 dtype으로 바꿔 device에 올립니다."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as file:
        small = {k: file.get_tensor(names[k]) for k in names if k.endswith(QUANT_SUFFIXES)}
        prefixes = {k.rsplit(".", 1)[0] + "." for k in small}
        for prefix in prefixes:
            check_int8(small, prefix)
        replaced = replace_linears_with_int8(model, {p: convrot_group(small, p) for p in prefixes})
        if missing := prefixes - {f"{n}." for n in replaced}:
            raise ValueError(f"{path.name}: INT8 층과 맞는 선형 층이 없습니다: {sorted(missing)[:8]}")
        wanted = set(model.state_dict())
        have = {k for k in names if not k.endswith(QUANT_SUFFIXES)}
        if wanted != have:
            raise ValueError(f"{path.name}: 키가 다릅니다. 없는 키 {sorted(wanted - have)[:8]}, 남는 키 {sorted(have - wanted)[:8]}")
        state = {}
        for key in have:
            tensor = file.get_tensor(names[key])
            is_int8 = key.endswith(".weight") and key[:-len("weight")] in prefixes
            state[key] = tensor.to(device) if is_int8 else tensor.to(device, dtype)
    model.load_state_dict(state, strict=True, assign=True)
    for name in replaced:
        model.get_submodule(name).scale = layer_scale(small, f"{name}.").to(device)


def check_names(model: nn.Module, header: dict, names: dict[str, str], squeeze_time: bool = False) -> dict[str, str]:
    """모델과 파일의 키, 모양이 정확히 맞는지 확인합니다."""
    state = model.state_dict()
    wanted = set(state)
    supplied = {key for key in names if not key.endswith(QUANT_SUFFIXES)}
    if wanted != supplied:
        raise ValueError(f"체크포인트 구조가 다릅니다: 없는 키 {sorted(wanted - supplied)[:8]}, 남는 키 {sorted(supplied - wanted)[:8]}")
    unused = set(header) - set(names.values()) - {".comfy_attention.config"}
    if unused:
        raise ValueError(f"쓰이지 않는 체크포인트 키: {sorted(unused)[:8]}")
    for key in wanted:
        shape = header[names[key]]["shape"]
        if squeeze_time and len(shape) == 5 and shape[2] == 1:
            shape = shape[:2] + shape[3:]
        if list(state[key].shape) != shape:
            raise ValueError(f"체크포인트 모양이 다릅니다: {key}")
    return names


def remove_loras(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, Int8Linear):
            module.loras.clear()


def attach_lora(model: nn.Module, path: Path, strength: float) -> None:
    """기존 LoRA를 모두 떼고 path의 LoRA를 strength 세기로 붙입니다."""
    from safetensors.torch import load_file

    remove_loras(model)
    weights = load_file(str(path))
    used, terms = set(), []
    for key in weights:
        suffix = next((s for s in (".lora_A.weight", ".lora_A.default.weight", ".lora_down.weight") if key.endswith(s)), None)
        if suffix is None:
            continue
        prefix = key[:-len(suffix)]
        name = prefix.removeprefix("diffusion_model.").removeprefix("model.diffusion_model.")
        module = model.get_submodule(name)
        if not isinstance(module, Int8Linear):
            raise ValueError(f"LoRA 대상이 INT8 선형 층이 아닙니다: {name}")
        up_key = prefix + suffix.replace("lora_A", "lora_B").replace("lora_down", "lora_up")
        alpha_key = prefix + ".alpha"
        down, up = weights[key], weights[up_key]
        rank = down.shape[0]
        if down.shape[1] != module.in_features or up.shape != (module.out_features, rank):
            raise ValueError(f"LoRA 모양이 다릅니다: {name}")
        scale = strength * (float(weights[alpha_key]) / rank if alpha_key in weights else 1)
        device = module.weight.device
        terms.append((module, (down.to(device), up.to(device), scale)))
        used.update((key, up_key))
        if alpha_key in weights:
            used.add(alpha_key)
    if used != set(weights) or not terms:
        raise ValueError(f"지원하지 않는 LoRA 구조입니다: {sorted(set(weights) - used)[:8]}")
    for module, term in terms:
        module.loras.append(term)
