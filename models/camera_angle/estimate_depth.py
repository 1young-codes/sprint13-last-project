"""MoGe로 사진 한 장의 3D 점 구름과 원본 화각을 추정합니다 (그래픽 카드 필요).

불러오는 방식은 ai-generation 프로젝트의 backend/studio/engine/moge3.py와 같습니다.
가중치는 Comfy-Org/MoGe의 fp16 파일이고, 추론 코드는 microsoft/MoGe 고정 커밋을 download_models.py가 받아 둡니다.
"""

import json
import re
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .settings import code_folder, load_settings, model_file


@dataclass
class DepthResult:
    points: np.ndarray  # (H, W, 3) 카메라 기준 좌표, x 오른쪽, y 위, z 앞
    valid: np.ndarray   # (H, W) 믿을 수 있는 점
    fov_deg: float      # 추정한 가로 화각


def rename_moge3_weights(weights: dict) -> dict:
    """Comfy-Org 파일의 이름표(transformers 형식)를 MoGe 원본 코드의 이름표로 바꿉니다."""
    prefix = "encoder.backbone."
    renamed = {}
    for key, value in weights.items():
        if key.startswith(prefix):
            key = key[len(prefix):]
            key = key.replace("embeddings.patch_embeddings.projection", "patch_embed.proj")
            key = key.replace("embeddings.position_embeddings", "pos_embed").removeprefix("embeddings.")
            key = key.replace("encoder.layer.", "blocks.").replace("layernorm.", "norm.")
            key = key.replace("attention.output.dense", "attn.proj").replace("layer_scale1.lambda1", "ls1.gamma")
            key = (key.replace("layer_scale2.lambda1", "ls2.gamma")
                   .replace("mlp.weights_in", "mlp.w12").replace("mlp.weights_out", "mlp.w3"))
            if match := re.fullmatch(r"blocks\.(\d+)\.attention\.attention\.(query|key|value)\.(weight|bias)", key):
                layer, component, param = match.groups()
                if component != "query":
                    continue
                stem = f"{prefix}encoder.layer.{layer}.attention.attention."
                value = torch.cat([weights[f"{stem}{part}.{param}"] for part in ("query", "key", "value")])
                key = f"blocks.{layer}.attn.qkv.{param}"
            key = prefix + key
        renamed[key] = value
    return renamed


def load_depth_model(model_name: str):
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    code = code_folder("moge")
    weights = model_file(model_name)
    if not weights.is_file() or not code.is_dir():
        raise FileNotFoundError(f"깊이 모델 파일이 없습니다: {weights}, {code}. "
                                "먼저 python -m models.camera_angle.download_models 를 실행하세요")
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    if model_name == "moge2":
        from moge.model.v2 import MoGeModel
        config = json.loads((code / "configs/train/v2.json").read_text(encoding="utf-8"))["model"]
        with init_empty_weights():
            model = MoGeModel(**config)
        model.load_state_dict(load_file(str(weights)), strict=True, assign=True)
        return model.eval().requires_grad_(False).float().to("cuda")
    from moge.model.v3 import MoGeModel
    config = json.loads((code / "configs/train/v3.json").read_text(encoding="utf-8"))["model"]
    # Comfy-Org 파일은 ViT-G 백본이라 설정의 차원을 그에 맞춥니다 (ai-generation에서 검증한 값)
    config["encoder"].update(backbone="dinov2_vitg14", intermediate_layers=[9, 19, 29, 39], dim_out=1536)
    config["neck"]["dim_in"][0] = config["refiner"]["encoder_channels"] = 1538
    widths = [1024, 512, 256, 128, 64]
    for name in ("neck", "points_head", "mask_head", "normal_head"):
        config[name]["dim_res_blocks"] = widths
        if name != "neck":
            config[name]["dim_in"] = widths
    config["scale_head"]["dims"] = [1536, 1536, 1536, 1]
    with init_empty_weights():
        model = MoGeModel(**config)
    model.load_state_dict(rename_moge3_weights(load_file(str(weights))), strict=True, assign=True)
    # 복셀 좌표는 FP32가 필요하고, 학습된 특징 투영은 FP16을 받습니다
    model.refiner.input_proj.register_forward_pre_hook(lambda module, args: (args[0].to(module.weight.dtype),))
    return model.eval().requires_grad_(False).to("cuda")


@torch.inference_mode()
def estimate_depth_and_fov(photo: Image.Image) -> DepthResult:
    """모델을 불러와 한 번 추론하고, 그래픽 메모리를 비운 뒤 결과를 돌려줍니다."""
    depth = load_settings()["depth"]
    model = load_depth_model(depth["model"])
    try:
        rgb = np.array(photo.convert("RGB"))
        source = torch.from_numpy(rgb).permute(2, 0, 1).to("cuda", dtype=torch.float32) / 255
        options = dict(resolution_level=depth["resolution_level"], apply_mask=True, force_projection=True)
        if depth["model"] == "moge3":
            options["refine_steps"] = depth["refine_steps"]
        with torch.autocast("cuda", dtype=torch.float16):
            geometry = model.infer(source, **options)
        points = geometry["points"].float().cpu().numpy()
        valid = np.isfinite(points).all(-1) & (points[..., 2] > 0) & geometry["mask"].cpu().numpy().astype(bool)
        focal = float(geometry["intrinsics"].float().cpu().numpy()[0, 0])  # 가로 길이로 나눈 초점 거리
    finally:
        del model
        torch.cuda.empty_cache()
    points = np.nan_to_num(points) * np.array([1, -1, 1], dtype=np.float32)  # MoGe는 y가 아래쪽
    return DepthResult(points=points, valid=valid, fov_deg=float(np.degrees(2 * np.arctan(0.5 / focal))))
