"""VLM이 짚은 배경 잡동사니를 LaMa로 지웁니다. 상품 영역(마스크)은 절대 건드리지 않습니다."""

import numpy as np
from scipy.ndimage import binary_dilation

from .download_models import model_folder


def resize(array: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    from PIL import Image

    mode = "F" if array.ndim == 2 else None
    if mode:
        return np.asarray(Image.fromarray(array.astype(np.float32), mode).resize(size, Image.NEAREST if nearest else Image.BILINEAR))
    picture = Image.fromarray((array * 255).round().astype(np.uint8)).resize(size, Image.BICUBIC)
    return np.asarray(picture, np.float32) / 255


def box_pixels(box: list[float], width: int, height: int, grow: float) -> tuple[int, int, int, int]:
    """0~1000 상자를 grow배 여유를 둔 화소 좌표로 (사진 밖은 잘라 냄)."""
    x0, y0, x1, y1 = box[0] * width / 1000, box[1] * height / 1000, box[2] * width / 1000, box[3] * height / 1000
    pad = grow * max(x1 - x0, y1 - y0)
    return (max(int(x0 - pad), 0), max(int(y0 - pad), 0), min(int(np.ceil(x1 + pad)), width), min(int(np.ceil(y1 + pad)), height))


def inpaint(model, crop: np.ndarray, hole: np.ndarray, long_side: int) -> np.ndarray:
    import torch

    height, width = crop.shape[:2]
    scale = min(1.0, long_side / max(height, width))
    size = (max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8))  # LaMa는 8의 배수
    image = torch.from_numpy(resize(crop, size)).permute(2, 0, 1)[None].cuda()
    mask = torch.from_numpy((resize(hole.astype(np.float32), size, nearest=True) > 0.5).astype(np.float32))[None, None].cuda()
    with torch.no_grad():
        filled = model(image, mask)[0].permute(1, 2, 0).float().cpu().numpy().clip(0, 1)
    return resize(filled, (width, height))


def remove_distractions(image: np.ndarray, distractions: list[dict], subject: np.ndarray, rule: dict) -> tuple[np.ndarray, list[str]]:
    import torch

    height, width = image.shape[:2]
    keep = binary_dilation(subject > rule["subject_threshold"], iterations=rule["subject_margin_px"])
    model = torch.jit.load(str(model_folder("inpaint") / "lama.pt"), map_location="cuda").eval()
    result, removed = image.copy(), []
    for item in distractions:
        x0, y0, x1, y1 = box_pixels(item["box"], width, height, rule["box_padding"])
        if (x1 - x0) * (y1 - y0) > rule["max_box_share"] * width * height:
            continue
        hole = np.zeros((height, width), bool)
        hole[y0:y1, x0:x1] = True
        hole &= ~keep
        if not hole.any():
            continue
        cx0, cy0, cx1, cy1 = box_pixels(item["box"], width, height, rule["context_scale"])
        crop, crop_hole = result[cy0:cy1, cx0:cx1], hole[cy0:cy1, cx0:cx1]
        filled = inpaint(model, crop, crop_hole, rule["crop_long_side"])
        result[cy0:cy1, cx0:cx1] = np.where(crop_hole[..., None], filled, crop)
        removed.append(item["label"])
    del model
    torch.cuda.empty_cache()
    return result, removed
