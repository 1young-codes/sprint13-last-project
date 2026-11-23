"""360도 배경의 왼쪽 끝과 오른쪽 끝이 만나는 세로 이음매를 고칩니다.

빈 곳 채우기는 양 끝이 이어진다는 것을 보장하지 않아, 뒤쪽을 보는 프리셋(reverse 등)에 이음매가 보입니다.
원본 저장소의 Seam Prep 방식처럼 배경을 반 바퀴 돌려 이음매를 가운데로 옮기고, 가운데 띠만 다시 그린 뒤 되돌립니다.
"""

import numpy as np
from PIL import Image, ImageFilter

from .settings import load_settings


def center_band_mask(width: int, height: int, band_px: int, blur_px: int) -> Image.Image:
    """가운데 세로 띠가 흰색인 마스크. 경계는 blur_px만큼 부드럽게 합니다."""
    columns = np.zeros(width, dtype=np.uint8)
    columns[width // 2 - band_px // 2:width // 2 + (band_px + 1) // 2] = 255
    mask = Image.fromarray(np.broadcast_to(columns, (height, width)).copy())
    return mask.filter(ImageFilter.GaussianBlur(blur_px / 2)) if blur_px else mask


def repair_left_right_seam(model, view_360: Image.Image, seed: int, steps: int) -> Image.Image:
    settings = load_settings()["view_360"]
    shift = view_360.width // 2
    rolled = Image.fromarray(np.roll(np.asarray(view_360.convert("RGB")), shift, axis=1))
    model.use_lora(None, 0)  # 일반 편집 (LoRA 없음)
    redrawn = model.edit(rolled, settings["seam_prompt"], seed, steps).resize(rolled.size, Image.Resampling.LANCZOS)
    mask = center_band_mask(*rolled.size, settings["seam_band_px"], settings["seam_blur_px"])
    repaired = Image.composite(redrawn, rolled, mask)
    return Image.fromarray(np.roll(np.asarray(repaired), -shift, axis=1))
