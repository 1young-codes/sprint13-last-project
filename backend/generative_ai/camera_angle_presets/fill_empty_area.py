"""다시 찍은 사진에서 원본이 찍지 않은 빈 곳(표시색)을 생성 모델로 채웁니다."""

import numpy as np
from PIL import Image, ImageFilter

from .camera_math import to_uint8_image
from .settings import load_settings


def box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    """가로세로 (2 × radius + 1) 상자 평균. 바깥은 0으로 봅니다. 세 번 겹치면 가우시안 흐림과 비슷합니다."""
    for axis in (0, 1):
        padded = np.pad(values, [(radius + 1, radius) if a == axis else (0, 0) for a in range(values.ndim)])
        total = np.cumsum(padded, axis=axis)
        values = (np.take(total, range(2 * radius + 1, total.shape[axis]), axis=axis)
                  - np.take(total, range(0, total.shape[axis] - 2 * radius - 1), axis=axis)) / (2 * radius + 1)
    return values


def match_colors_to_original(generated: np.ndarray, original: np.ndarray, known: np.ndarray,
                             radius: int) -> np.ndarray:
    """생성 결과의 밝기와 색을 원본에 맞춥니다. 생성 모델은 원본 부분도 조금 다른 색으로 다시 그리므로,
    그 차이(원본 - 생성)를 원본이 있는 곳에서 재서 흐리게 번지게 하고, 원본 가까운 생성 영역에 더합니다.
    원본에서 radius보다 먼 곳은 차이를 알 수 없어 점점 덜 고칩니다. 배열은 모두 0~1 실수입니다."""
    difference, weight = (original - generated) * known[..., None], known
    for _ in range(3):
        difference, weight = box_blur(difference, radius), box_blur(weight, radius)
    offset = difference / np.maximum(weight, 1e-4)[..., None]
    reach = np.clip(weight / weight.max(initial=1e-4) * 4, 0, 1)[..., None]  # 원본에서 멀어질수록 0
    return np.clip(generated + offset * reach, 0, 1)


def fill_empty_area(model, image: Image.Image, empty_mask: Image.Image, scene_description: str,
                    seed: int, steps: int) -> Image.Image:
    """빈 곳만 생성 결과로 바꾸고, 원본 픽셀은 그대로 둡니다. 빈 곳이 없으면 생성하지 않습니다."""
    settings = load_settings()["rotate"]
    if not empty_mask.getbbox():
        return image
    model.use_lora(None, 0)
    prompt = settings["fill_prompt"].format(scene=scene_description).strip()
    filled = model.edit(image, prompt, seed, steps).resize(image.size, Image.Resampling.LANCZOS)
    empty = np.asarray(empty_mask.convert("L"), dtype=np.float32) / 255
    original = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    generated = np.asarray(filled.convert("RGB"), dtype=np.float32) / 255
    radius = max(1, round(settings["color_match_ratio"] * image.width / 3))  # 상자 세 번 ≈ 이 폭의 흐림
    filled = to_uint8_image(match_colors_to_original(generated, original, (empty < .5).astype(np.float32), radius))
    # 빈 곳을 조금 넓힌 뒤 흐리게 해서, 원본과 생성 영역의 경계선이 보이지 않게 섞습니다
    feather = settings["feather_px"]
    mask = empty_mask.convert("L")
    if feather > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * feather + 1)).filter(ImageFilter.GaussianBlur(feather / 2))
    return Image.composite(filled, image.convert("RGB"), mask)
