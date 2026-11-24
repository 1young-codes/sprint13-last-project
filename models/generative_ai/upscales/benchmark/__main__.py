"""업스케일 모델 비교를 실행합니다. benchmark.toml의 조건마다 모든 모델로 업스케일하고, 지표를 재고, 모델 하나를 고릅니다.

    python -m models.generative_ai.upscales.benchmark                                  # 전부
    python -m models.generative_ai.upscales.benchmark --model vosr --condition x4_real   # 일부만
    python -m models.generative_ai.upscales.benchmark --skip-run                        # 이미 만든 결과로 지표와 보고서만

이미 만든 결과와 지표는 다시 만들지 않습니다. 다시 만들려면 --force를 붙입니다.
결과: settings.toml output_dir/benchmark/ (report.md, summary.json, 조건별 확대 비교 그림)
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from ..run_model import run
from ..settings import PACKAGE_DIR, PROJECT_ROOT, load_settings, model_names, project_path, read_toml
from .report import average, contact_sheet, select, write_report

CONFIG = read_toml(PACKAGE_DIR / "benchmark" / "benchmark.toml")


def gather(spec: dict) -> list[tuple[str, Path]]:
    """조건에 쓸 사진들. 하위 폴더에 있으면 '폴더__이름'으로 부릅니다."""
    root = project_path(spec["source"])
    suffixes = load_settings()["run"]["image_suffixes"]
    found = [p for p in sorted(root.glob(spec["pattern"]))
             if p.suffix.lower() in suffixes and p.stem in spec.get("names", [p.stem])]
    if not found:
        raise FileNotFoundError(f"{root}에 '{spec['pattern']}' 사진이 없습니다 (settings.toml [paths].{spec['source']})")
    return [("__".join(p.relative_to(root).with_suffix("").parts), p) for p in found]


def prepare(spec: dict, folder: Path) -> list[dict]:
    """모델 입력(한 장씩 폴더 하나)과 정답 사진을 만듭니다."""
    scale = spec["scale"]
    items = []
    for name, source in gather(spec):
        low_path = folder / "inputs" / name / f"{name}.png"
        truth_path = folder / "truth" / f"{name}.png" if spec.get("degrade") else None
        if not low_path.is_file():
            low_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image = image.convert("RGB")
                if truth_path:  # 정답은 배율로 나눠떨어지게 자르고, 입력은 그걸 1/scale로 줄인 것
                    image = image.crop((0, 0, image.width // scale * scale, image.height // scale * scale))
                    truth_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(truth_path)
                    image = image.resize((image.width // scale, image.height // scale), Image.BICUBIC)
                image.save(low_path)
        items.append({"name": name, "input": low_path, "truth": truth_path})
    return items


def upscale(model: str, item: dict, scale: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if model == CONFIG["baseline"]:
        start = time.perf_counter()
        with Image.open(item["input"]) as image:
            image.resize((image.width * scale, image.height * scale), Image.BICUBIC).save(out / f"{item['name']}.png")
        info = {"returncode": 0, "seconds_per_image": time.perf_counter() - start, "peak_vram_mb": 0,
                "outputs": {item["name"]: {"status": "ok"}}}
        (out / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        return
    try:
        info = run(model, item["input"].parent, out, scale)
        status = info["outputs"].get(item["name"], {}).get("status", f"실패 (종료 코드 {info['returncode']})")
    except Exception as error:  # 한 모델이 실패해도 나머지 비교는 계속합니다
        (out / "run_info.json").write_text(json.dumps({"error": str(error)}, ensure_ascii=False), encoding="utf-8")
        status = f"실패: {error}"
    print(f"  {model} {item['name']}: {status}")


def score(scorer_box: list, model: str, item: dict, out: Path, force: bool) -> dict:
    row = {"model": model, "image": item["name"]}
    info_path, result_path, cache = out / "run_info.json", out / f"{item['name']}.png", out / "scores.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    row |= {k: info[k] for k in ("seconds_per_image", "peak_vram_mb") if k in info}
    if not result_path.is_file():
        return row | {"status": "missing"}
    if cache.is_file() and not force:
        return row | {"status": "ok"} | json.loads(cache.read_text(encoding="utf-8"))
    if not scorer_box:
        from .metrics import Scorer

        scorer_box.append(Scorer(CONFIG["metrics"], project_path("metric_models_dir")))
    regions = CONFIG.get("regions", {}).get(item["name"], {})
    with Image.open(item["input"]) as low, Image.open(result_path) as result:
        truth = Image.open(item["truth"]).convert("RGB") if item["truth"] else None
        scores = scorer_box[0].score(low.convert("RGB"), result.convert("RGB"), truth, regions)
    cache.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return row | {"status": "ok"} | scores


def lower_better_table() -> dict:
    from pyiqa.default_model_configs import DEFAULT_CONFIGS

    names = CONFIG["metrics"]["full_reference"] + CONFIG["metrics"]["no_reference"]
    return {n: bool(DEFAULT_CONFIGS.get(n, {}).get("lower_better", False)) for n in names}


def main():
    parser = argparse.ArgumentParser(prog="python -m models.generative_ai.upscales.benchmark", description="업스케일 모델을 비교합니다")
    parser.add_argument("--model", action="append", help="비교할 모델 (여러 번 지정 가능, 기본: 전부)")
    parser.add_argument("--condition", action="append", help=f"조건 (기본: 전부) {list(CONFIG['conditions'])}")
    parser.add_argument("--skip-run", action="store_true", help="업스케일은 건너뛰고 지표와 보고서만")
    parser.add_argument("--force", action="store_true", help="이미 있는 결과와 지표도 다시 만듭니다")
    args = parser.parse_args()

    models = [CONFIG["baseline"], *(args.model or model_names())]
    conditions = args.condition or list(CONFIG["conditions"])
    root = project_path("output_dir") / "benchmark"
    plans = {c: prepare(CONFIG["conditions"][c], root / c) for c in conditions}

    if not args.skip_run:
        for cond, items in plans.items():
            scale = CONFIG["conditions"][cond]["scale"]
            print(f"[{cond}] 사진 {len(items)}장 × 배율 {scale}")
            for model in models:
                for item in items:
                    out = root / cond / model / item["name"]
                    if args.force or not (out / f"{item['name']}.png").is_file():
                        upscale(model, item, scale, out)

    rows, scorer_box = [], []  # 지표 모델은 업스케일 모델이 그래픽 메모리를 다 비운 뒤에 불러옵니다
    for cond, items in plans.items():
        for model in models:
            for item in items:
                rows.append({"condition": cond} | score(scorer_box, model, item, root / cond / model / item["name"], args.force))

    for cond, items in plans.items():
        scale = CONFIG["conditions"][cond]["scale"]
        for item in items:
            regions = CONFIG.get("regions", {}).get(item["name"])
            if regions:
                results = {m: root / cond / m / item["name"] / f"{item['name']}.png" for m in models}
                contact_sheet(results, regions, scale, CONFIG["report"]["crop_height"], root / f"sheet_{cond}_{item['name']}.png")

    means = average(rows)
    choice = select(means, models[1:], CONFIG["selection"], lower_better_table())
    path = write_report(root, CONFIG, means, choice, models)
    print(f"보고서: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
    print(f"선정: {choice['chosen']}")


if __name__ == "__main__":
    main()
