"""원본과 결과 사진 쌍을 비교해, 상품이 원본과 달라지지 않았는지 숫자로 보여줍니다.

보정, 업스케일처럼 구도가 그대로인 결과용입니다. 각도를 바꾼 사진은 화소 위치가 달라 이 지표가 맞지 않습니다.

- iou: 원본과 결과에서 각각 잡은 상품 영역이 얼마나 겹치는지 (1이면 모양과 위치 그대로)
- structure: 상품 영역 밝기의 SSIM 구조 항 (무늬나 글자가 새로 그려지거나 지워지면 떨어짐)
- delta_e: 상품 영역 평균 색 차이, CIELAB ΔE76 (클수록 색이 많이 바뀜)

색 차이와 구조 계산은 보정 기능의 실물 검사(photo_retouch/fidelity.py)를 그대로 써서 기준이 같습니다.

예 (VM, 상품 영역 모델이 그래픽 카드를 씀):
    python -m evaluation.product_fidelity pairs.csv --out /data/outputs/na_sanghun/fidelity.csv
pairs.csv는 한 줄에 "원본 경로,결과 경로" 입니다. 결과가 원본보다 크면(업스케일) 원본 크기로 줄여 비교합니다.
"""

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from models.generative_ai.photo_retouch.adjustments import luminance
from models.generative_ai.photo_retouch.fidelity import structure_map, to_lab
from models.generative_ai.photo_retouch.settings import load_settings

METRICS = ("iou", "structure", "delta_e")


def load(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, np.float32) / 255


def score(before: np.ndarray, after: np.ndarray, mask_before: np.ndarray, mask_after: np.ndarray) -> dict:
    """사진은 0~1 RGB, 마스크는 0~1. 색과 구조는 원본의 상품 영역에서 잽니다."""
    a, b = mask_before > 0.5, mask_after > 0.5
    union = (a | b).sum()
    region = a if a.any() else np.ones_like(a)  # 상품을 못 잡으면 사진 전체
    sigma = load_settings()["fidelity"]["structure_sigma_px"]
    return {
        "iou": float((a & b).sum() / union) if union else 1.0,
        "structure": float(structure_map(luminance(before), luminance(after), sigma)[region].mean()),
        "delta_e": float(np.linalg.norm(to_lab(after) - to_lab(before), axis=-1)[region].mean()),
    }


def summarize(rows: list[dict]) -> dict:
    """지표마다 중앙값과 가장 나쁜 값. iou와 structure는 작을수록, delta_e는 클수록 나쁩니다."""
    out = {}
    for m in METRICS:
        values = [r[m] for r in rows]
        out[m] = {"median": statistics.median(values), "worst": max(values) if m == "delta_e" else min(values)}
    return out


def load_mask_fn():
    from models.generative_ai.photo_retouch.product_mask import load_mask_model, product_mask

    model = load_mask_model()
    return lambda image: product_mask(model, image)


def evaluate(pairs: list[tuple[Path, Path]], mask_fn) -> list[dict]:
    rows = []
    for original, result in pairs:
        before = load(original)
        after = load(result, size=(before.shape[1], before.shape[0]))
        rows.append({"original": str(original), "result": str(result),
                     **score(before, after, mask_fn(before), mask_fn(after))})
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pairs", type=Path, help='한 줄에 "원본 경로,결과 경로"인 CSV')
    p.add_argument("--out", type=Path, required=True, help="사진별 지표를 쓸 CSV")
    args = p.parse_args(argv)

    with open(args.pairs, encoding="utf-8") as f:
        pairs = [(Path(r[0].strip()), Path(r[1].strip())) for r in csv.reader(f) if r and not r[0].startswith("#")]
    rows = evaluate(pairs, load_mask_fn())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["original", "result", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)}쌍 → {args.out}")
    for m, s in summarize(rows).items():
        print(f"{m:10s} 중앙값 {s['median']:.3f}  가장 나쁜 값 {s['worst']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
