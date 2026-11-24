"""사진을 재서 기본 보정 값을 정합니다 (노출, 색 치우침, 대비). 사진 해석 없이 도는 기준 후보입니다."""

import numpy as np

from .adjustments import clamp_params, luminance, srgb_to_linear


def auto_params(image: np.ndarray, auto: dict, ops: dict, bounds: dict, mask: np.ndarray | None = None) -> tuple[dict, dict]:
    """(보정 값, 값별 한국어 이유). mask가 있으면 노출은 상품 영역 밝기로 정합니다."""
    lum = luminance(image)
    subject = lum[mask > 0.5] if mask is not None and (mask > 0.5).any() else lum
    usable = (lum > 0.05) & (lum < 0.95)  # 날아간 곳과 너무 어두운 곳은 색 치우침 계산에서 뺌
    params, reasons = {}, {}

    median = float(np.median(subject))
    target = float(np.clip(median, *auto["mid_gray_band"]))
    ev = float(np.log2(srgb_to_linear(np.float32(target)) / max(srgb_to_linear(np.float32(median)), 1e-4)))
    ev = float(np.clip(ev, -auto["max_auto_ev"], auto["max_auto_ev"]))
    if abs(ev) >= 0.1:
        params["exposure_ev"] = ev
        reasons["exposure_ev"] = f"사진 밝기 중앙값이 {median:.2f}라 {'밝게' if ev > 0 else '어둡게'} 맞췄습니다"

    if usable.mean() > 0.05:
        r, g, b = srgb_to_linear(image[usable]).mean(axis=0)
        strength = auto["white_balance_strength"]
        temperature = strength * (b - r) / (ops["white_balance_gain"] * (r + b))
        tint = strength * (1 - (r + b) / (2 * g)) / ops["tint_gain"]
        if abs(temperature) >= 0.05:
            params["temperature"] = float(temperature)
            reasons["temperature"] = f"사진 전체가 {'푸르게' if temperature > 0 else '노랗게'} 치우쳐 색온도를 맞췄습니다"
        if abs(tint) >= 0.05:
            params["tint"] = float(tint)
            reasons["tint"] = f"사진 전체가 {'초록' if tint > 0 else '자홍'}빛으로 치우쳐 바로잡았습니다"

    low, high = np.percentile(lum, [auto["low_percentile"], auto["high_percentile"]])
    spread = float(high - low)
    if spread < auto["target_spread"]:
        params["contrast"] = float((auto["target_spread"] - spread) * auto["contrast_per_spread"])
        reasons["contrast"] = f"밝은 곳과 어두운 곳 차이가 작아({spread:.2f}) 대비를 올렸습니다"

    return clamp_params(params, bounds), reasons
