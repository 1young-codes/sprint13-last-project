"""사진을 재서 바로잡을 거리를 찾고, 각각의 보정 값을 계산합니다.

값은 여기서만 정합니다. 해석 모델은 이 중 무엇이 진짜 문제인지(일부러 낸 분위기가 아닌지)만 판단합니다.
결과: {이름: {"params": 보정 값, "reason": 한국어 이유, "finding": 해석 모델에게 보여 줄 영어 측정 설명}}
"""

import numpy as np

from .adjustments import clamp_params, luminance, srgb_to_linear


def measure(image: np.ndarray, auto: dict, ops: dict, bounds: dict, mask: np.ndarray | None = None) -> dict:
    lum = luminance(image)
    inside = mask > 0.5 if mask is not None and (mask > 0.5).any() else np.ones(lum.shape, bool)
    subject = lum[inside]
    found = {}

    def add(name, params, reason, finding):
        params = clamp_params(params, bounds)
        if params:
            found[name] = {"params": params, "reason": reason, "finding": finding}

    median = float(np.median(subject))
    target = float(np.clip(median, *auto["mid_gray_band"]))
    ev = float(np.log2(srgb_to_linear(np.float32(target)) / max(srgb_to_linear(np.float32(median)), 1e-4)))
    ev = float(np.clip(ev, -auto["max_auto_ev"], auto["max_auto_ev"]))
    if abs(ev) >= auto["min_ev"]:
        add("exposure", {"exposure_ev": ev}, f"상품 밝기가 {'어두워' if ev > 0 else '너무 밝아'} {'밝게' if ev > 0 else '어둡게'} 맞췄습니다",
            f"subject median brightness {median:.2f}, would {'brighten' if ev > 0 else 'darken'} by {abs(ev):.2f} stops")

    # 색 치우침은 원래 무채색이어야 할 화소(흰색·회색에 가까운 곳)로만 잼. 사진 전체 평균으로 재면
    # 하늘이 많은 사진은 "푸르다", 주황 옷이 많은 사진은 "노랗다"고 잘못 잼. 그런 화소가 적으면 색은 건드리지 않음
    peak = image.max(axis=-1)
    chroma = (peak - image.min(axis=-1)) / np.maximum(peak, 1e-4)
    low, high = auto["neutral_brightness"]
    usable = (lum > low) & (lum < high) & (chroma < auto["neutral_max_chroma"])
    if usable.mean() > auto["min_neutral_share"]:
        r, g, b = srgb_to_linear(image[usable]).mean(axis=0)
        strength = auto["white_balance_strength"]
        temperature = strength * (b - r) / (ops["white_balance_gain"] * (r + b))
        tint = strength * (1 - (r + b) / (2 * g)) / ops["tint_gain"]
        params = {k: float(v) for k, v in (("temperature", temperature), ("tint", tint)) if abs(v) >= auto["min_color_shift"]}
        if params:
            cast = ("blue" if temperature > 0 else "yellow") if "temperature" in params else ("green" if tint > 0 else "magenta")
            add("white_balance", params, f"사진 전체가 {'푸르게' if cast == 'blue' else '노랗게' if cast == 'yellow' else '초록빛으로' if cast == 'green' else '자홍빛으로'} 치우쳐 바로잡았습니다",
                f"overall {cast} color cast, would neutralize it")

    low, high = np.percentile(lum, [auto["low_percentile"], auto["high_percentile"]])
    spread = float(high - low)
    if spread < auto["target_spread"]:
        add("contrast", {"contrast": (auto["target_spread"] - spread) * auto["contrast_per_spread"]},
            "사진이 뿌옇게 보여 대비를 조금 올렸습니다", f"tonal range only {spread:.2f} of full range, looks flat or hazy")

    bright = float((subject > auto["highlight_level"]).mean())
    if bright > auto["highlight_share"]:
        add("highlights", {"highlights": -auto["highlight_strength"]}, "상품의 밝은 곳이 하얗게 날아가려 해 눌렀습니다",
            f"{bright:.0%} of the subject is near white")
    dark = float((subject < auto["shadow_level"]).mean())
    if dark > auto["shadow_share"]:
        add("shadows", {"shadows": auto["shadow_strength"]}, "상품의 어두운 곳이 뭉개져 보여 살짝 밝혔습니다",
            f"{dark:.0%} of the subject is near black")
    return found


def combine(found: dict, names) -> tuple[dict, dict]:
    """고른 바로잡기들을 (보정 값, 값별 이유)로 합칩니다."""
    params, reasons = {}, {}
    for name in names:
        for key, value in found[name]["params"].items():
            params[key] = params.get(key, 0.0) + value
            reasons[key] = found[name]["reason"]
    return params, reasons
