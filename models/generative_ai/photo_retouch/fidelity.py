"""보정 전후의 상품 영역을 비교해, 실물과 달라지지 않았는지 검사합니다.

- 색 차이: CIELAB 평균 ΔE76 (ponytail: ΔE2000이 사람 눈에 더 가깝지만 통과선 용도로는 ΔE76으로 충분)
- 구조: 밝기의 SSIM 구조 항. 밝기와 대비가 바뀌어도 모양이 같으면 1에 가깝고, 무언가 그려지거나 지워지면 떨어집니다
- 날아감: 세부가 있던 화소가 새로 하얗게/까맣게 날아간 비율
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .adjustments import luminance, srgb_to_linear

D65 = np.array([0.95047, 1.0, 1.08883], np.float32)
RGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = srgb_to_linear(rgb) @ RGB_TO_XYZ.T / D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def structure_map(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    c3 = (0.03**2) / 2
    mu_a, mu_b = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    var_a = np.maximum(gaussian_filter(a * a, sigma) - mu_a**2, 0)
    var_b = np.maximum(gaussian_filter(b * b, sigma) - mu_b**2, 0)
    cov = gaussian_filter(a * b, sigma) - mu_a * mu_b
    return (cov + c3) / (np.sqrt(var_a * var_b) + c3)


def check(before: np.ndarray, after: np.ndarray, mask: np.ndarray | None, rule: dict) -> dict:
    region = np.ones(before.shape[:2], bool) if mask is None else mask > 0.5
    if region.mean() < rule["min_region_share"]:
        region = np.ones_like(region)
    delta_e = float(np.linalg.norm(to_lab(after) - to_lab(before), axis=-1)[region].mean())
    structure = float(structure_map(luminance(before), luminance(after), rule["structure_sigma_px"])[region].mean())
    # 세부가 있던 화소가 끝으로 날아간 것만 셈. 원래 거의 하얗던 반짝임이 끝에 닿는 건 잃은 세부가 없음.
    # 가장 밝은 채널로 재므로 진한 색에서 0인 채널은 뭉개짐으로 치지 않음
    level, margin = rule["clip_level"], rule["clip_margin"]
    peak_before, peak_after = before.max(axis=-1), after.max(axis=-1)
    blown = (peak_after >= level) & (peak_before < level - margin)
    crushed = (peak_after <= 1 - level) & (peak_before > 1 - level + margin)
    new_clipping = float((blown | crushed)[region].mean())

    problems = []
    if delta_e > rule["max_mean_delta_e"]:
        problems.append(f"상품 색이 너무 많이 바뀜 (ΔE {delta_e:.1f} > {rule['max_mean_delta_e']})")
    if structure < rule["min_structure"]:
        problems.append(f"상품 모양이 달라짐 (구조 {structure:.3f} < {rule['min_structure']})")
    if new_clipping > rule["max_new_clipping"]:
        problems.append(f"밝은 곳이나 어두운 곳이 날아감 ({new_clipping:.1%})")
    return {"ok": not problems, "delta_e": delta_e, "structure": structure, "new_clipping": new_clipping, "problems": problems}
