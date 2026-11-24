"""보정 값(노출, 색온도, 대비 ...)을 사진에 적용합니다. 픽셀을 새로 그리지 않고 밝기와 색만 바꿉니다.

사진은 0~1 실수 sRGB 배열(높이, 너비, 3)로 다룹니다. 노출과 색온도는 선형 빛 공간에서, 나머지는 sRGB에서 바꿉니다.
밝기 조정은 밝기 채널만 바꾼 뒤 RGB를 같은 비율로 곱해 색조가 돌지 않게 합니다.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def to_array(image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), np.float32) / 255


def to_image(array: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.clip(np.rint(array * 255), 0, 255).astype(np.uint8))


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def clamp_params(params: dict, bounds: dict) -> dict:
    """모르는 키는 버리고, 범위를 넘는 값은 잘라 냅니다."""
    clean = {}
    for key, value in params.items():
        if key in bounds and isinstance(value, (int, float)) and np.isfinite(value):
            clean[key] = float(np.clip(value, *bounds[key]))
    return clean


def scale_params(params: dict, strength: float) -> dict:
    return {key: value * strength for key, value in params.items()}


def add_params(*parts: dict) -> dict:
    total = {}
    for part in parts:
        for key, value in part.items():
            total[key] = total.get(key, 0.0) + value
    return total


def apply(image: np.ndarray, params: dict, ops: dict) -> np.ndarray:
    p = lambda key: params.get(key, 0.0)  # noqa: E731
    # 1) 노출, 색온도, 틴트: 선형 빛에서 곱하기
    light = srgb_to_linear(image) * 2.0 ** p("exposure_ev")
    gains = np.array([1 + ops["white_balance_gain"] * p("temperature"),
                      1 - ops["tint_gain"] * p("tint"),
                      1 - ops["white_balance_gain"] * p("temperature")], np.float32)
    rgb = linear_to_srgb(light * gains)

    # 2) 밝기 곡선: 대비, 밝은 곳, 어두운 곳 (끝점 0과 1은 움직이지 않음)
    lum = luminance(rgb)
    new = lum + p("contrast") * ops["contrast_gain"] * (lum - 0.5) * 4 * np.clip(lum * (1 - lum), 0, None)
    new = new + p("highlights") * ops["tone_gain"] * smoothstep(0.5, 1.2, new) * (new - 0.5)
    new = new + p("shadows") * ops["tone_gain"] * new * (1 - smoothstep(0.0, 0.6, new))

    # 3) 윤곽과 선명도: 밝기의 큰/작은 블러와의 차이를 더함
    if p("clarity"):
        sigma = ops["clarity_sigma"] * max(image.shape[:2])
        new = new + p("clarity") * ops["clarity_gain"] * (new - gaussian_filter(new, sigma)) * 4 * np.clip(new * (1 - new), 0, None)
    if p("sharpen"):
        new = new + p("sharpen") * ops["sharpen_gain"] * (new - gaussian_filter(new, ops["sharpen_sigma_px"]))
    rgb = rgb * (np.maximum(new, 0) / np.maximum(lum, 1e-4))[..., None]

    # 4) 채도와 생동감: 밝기는 두고 색 차이만 키움
    if p("saturation") or p("vibrance"):
        lum = luminance(rgb)[..., None]
        spread = np.clip(rgb.max(axis=-1, keepdims=True) - rgb.min(axis=-1, keepdims=True), 0, 1)
        factor = 1 + ops["saturation_gain"] * (p("saturation") + p("vibrance") * (1 - spread))
        rgb = lum + (rgb - lum) * factor
    return np.clip(rgb, 0, 1).astype(np.float32)


def apply_regions(image: np.ndarray, plan: dict, mask: np.ndarray | None, ops: dict) -> np.ndarray:
    """plan = {"global": {...}, "product": {...}, "background": {...}}. 상품·배경 값은 전체 값에 더해 마스크로 섞습니다."""
    whole = plan.get("global", {})
    if mask is None or not (plan.get("product") or plan.get("background")):
        return apply(image, whole, ops)
    product = apply(image, add_params(whole, plan.get("product", {})), ops)
    background = apply(image, add_params(whole, plan.get("background", {})), ops)
    weight = mask[..., None]
    return product * weight + background * (1 - weight)
