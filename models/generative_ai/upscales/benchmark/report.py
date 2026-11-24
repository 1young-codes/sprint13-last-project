"""지표를 모델별로 평균 내고, 통과선과 순위로 모델 하나를 고르고, 보고서와 확대 비교 그림을 만듭니다."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

ALWAYS_LOWER = {"seconds_per_image", "peak_vram_mb", "ocr_cer"}
ALWAYS_HIGHER = {"face_similarity"}


def is_lower_better(metric: str, lower_better: dict) -> bool:
    if metric in ALWAYS_LOWER or metric in ALWAYS_HIGHER:
        return metric in ALWAYS_LOWER
    base = metric.split("_", 1)[1] if metric.startswith(("lr_", "gt_")) else metric
    return lower_better.get(base, False)


def average(rows: list[dict]) -> dict:
    """(조건, 모델) -> 지표 평균. 값이 없거나 무한대인 사진은 빼고 냅니다."""
    groups = {}
    for row in rows:
        groups.setdefault((row["condition"], row["model"]), []).append(row)
    means = {}
    for key, group in groups.items():
        metrics = {k for row in group for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        means[key] = {}
        for metric in metrics:
            values = [row[metric] for row in group if isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])]
            if values:
                means[key][metric] = sum(values) / len(values)
        means[key]["images_ok"] = sum(row.get("status") == "ok" for row in group)
        means[key]["images"] = len(group)
    return means


def select(means: dict, candidates: list[str], selection: dict, lower_better: dict) -> dict:
    failed = {}
    for model in candidates:
        for gate in selection["gates"]:
            value = means.get((gate["condition"], model), {}).get(gate["metric"])
            if value is None:
                continue  # 이 조건에 해당 사진이 없음 (예: 얼굴 없는 사진만 있음)
            if ("min" in gate and value < gate["min"]) or ("max" in gate and value > gate["max"]):
                bound = f"≥ {gate['min']}" if "min" in gate else f"≤ {gate['max']}"
                failed.setdefault(model, []).append(f"{gate['condition']} {gate['metric']} = {value:.3f} (기준 {bound})")
        if any(v["images_ok"] < v["images"] for (_, m), v in means.items() if m == model):
            failed.setdefault(model, []).append("실패하거나 결과가 없는 사진이 있음")
    passed = [m for m in candidates if m not in failed]
    totals = {m: 0.0 for m in passed}
    for item in selection["rank"]:
        values = {m: means.get((item["condition"], m), {}).get(item["metric"]) for m in passed}
        present = sorted((v, m) for m, v in values.items() if v is not None)
        if not is_lower_better(item["metric"], lower_better):
            present.reverse()
        ranks = {m: i + 1 for i, (_, m) in enumerate(present)}
        for m in passed:  # 값이 없으면 꼴찌로 칩니다
            totals[m] += item["weight"] * ranks.get(m, len(passed))
    weight = sum(item["weight"] for item in selection["rank"]) or 1
    ranking = sorted(((m, t / weight) for m, t in totals.items()), key=lambda x: x[1])
    return {"failed": failed, "ranking": ranking, "chosen": ranking[0][0] if ranking else None}


def contact_sheet(result_dirs: dict, regions: dict, scale: int, height: int, path: Path) -> None:
    """영역마다 한 줄: 모델별 같은 영역을 같은 높이로 잘라 나란히 붙입니다."""
    rows = []
    for name, box in regions.items():
        tiles = []
        for model, image_path in result_dirs.items():
            if not image_path.is_file():
                continue
            with Image.open(image_path) as image:
                crop = image.convert("RGB").crop(tuple(v * scale for v in box))
            crop = crop.resize((max(1, round(crop.width * height / crop.height)), height), Image.LANCZOS)
            ImageDraw.Draw(crop).text((4, 4), f"{model} / {name}", fill=(255, 255, 0))
            tiles.append(crop)
        if tiles:
            row = Image.new("RGB", (sum(t.width for t in tiles), height))
            x = 0
            for tile in tiles:
                row.paste(tile, (x, 0))
                x += tile.width
            rows.append(row)
    if rows:
        sheet = Image.new("RGB", (max(r.width for r in rows), height * len(rows)))
        for i, row in enumerate(rows):
            sheet.paste(row, (0, i * height))
        sheet.save(path)


def fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}"


def write_report(folder: Path, config: dict, means: dict, choice: dict, models: list[str]) -> Path:
    lines = ["# 업스케일 모델 비교", ""]
    for cond, spec in config["conditions"].items():
        present = [m for m in models if (cond, m) in means]
        if not present:
            continue
        metrics = sorted({k for m in present for k in means[(cond, m)]} - {"images", "images_ok", "seconds", "returncode", "scale"})
        main = [k for k in metrics if not k.startswith("region_")]
        lines += [f"## {spec['label']} (`{cond}`)", "", "| 모델 | 사진 | " + " | ".join(main) + " |",
                  "| --- | --- |" + " --- |" * len(main)]
        for m in present:
            row = means[(cond, m)]
            lines.append(f"| {m} | {row['images_ok']}/{row['images']} | " + " | ".join(fmt(row.get(k)) for k in main) + " |")
        regional = [k for k in metrics if k.startswith("region_")]
        if regional:
            lines += ["", "영역별 되돌려 줄인 PSNR (높을수록 입력과 같음)", "",
                      "| 모델 | " + " | ".join(k.removeprefix("region_").removesuffix("_lr_psnr") for k in regional) + " |",
                      "| --- |" + " --- |" * len(regional)]
            lines += [f"| {m} | " + " | ".join(fmt(means[(cond, m)].get(k)) for k in regional) + " |" for m in present]
        lines.append("")
    lines += ["## 선정", ""]
    for model, reasons in choice["failed"].items():
        lines.append(f"- 탈락 `{model}`: " + "; ".join(reasons))
    lines += ["", "| 순위 | 모델 | 가중 평균 순위 |", "| --- | --- | --- |"]
    lines += [f"| {i} | {m} | {s:.2f} |" for i, (m, s) in enumerate(choice["ranking"], 1)]
    lines += ["", f"선정: **{choice['chosen'] or '없음 (모두 탈락)'}**", ""]
    path = folder / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(
        {"means": {f"{c}/{m}": v for (c, m), v in means.items()}, "selection": choice}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
