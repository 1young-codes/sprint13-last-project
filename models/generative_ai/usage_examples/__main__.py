"""활용 예시 이미지를 만듭니다. 원본 사진 → 해석(GPT) → MOSS 생성 → 각도 → 업스케일 → 보정.

사진 한 장을 끝까지:
    python -m models.generative_ai.usage_examples run --input 사진.jpg

비교 실험은 단계별로 (폴더 하나가 실험 하나):
    python -m models.generative_ai.usage_examples plan --input 사진폴더 --run 실험폴더
    python -m models.generative_ai.usage_examples calibrate --run 실험폴더
    python -m models.generative_ai.usage_examples generate --run 실험폴더 --category food
    python -m models.generative_ai.usage_examples generate --run 실험폴더 --category cosmetic household --condition original
    python -m models.generative_ai.usage_examples finish --run 실험폴더 --id food_korean_bibimbap
    python -m models.generative_ai.usage_examples review --run 실험폴더 --out 평가폴더
    python -m models.generative_ai.usage_examples summarize 순위_가.csv 순위_나.csv
"""

import argparse
import json
import tomllib
from pathlib import Path

from PIL import Image

from .settings import PROJECT_ROOT, load_settings, project_path

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_photos(path: Path) -> tuple[Path, list[dict]]:
    """사진 폴더(sources.toml이 있으면 그 목록과 묶음 이름) 또는 사진 한 장 → (폴더, [{id, file, group}])."""
    if path.is_file():
        return path.parent, [{"id": path.stem, "file": path.name, "group": ""}]
    sources = path / "sources.toml"
    if sources.is_file():
        with open(sources, "rb") as file:
            return path, [{"id": p["id"], "file": p["file"], "group": p["group"]} for p in tomllib.load(file)["photos"]]
    return path, [{"id": p.stem, "file": p.name, "group": ""} for p in sorted(path.iterdir()) if p.suffix.lower() in SUFFIXES]


def portable(folder: Path) -> str:
    """프로젝트 안이면 프로젝트 기준 상대 경로로 남겨, 다른 컴퓨터에서도 실험 폴더를 읽을 수 있게 합니다."""
    folder = folder.resolve()
    return folder.relative_to(PROJECT_ROOT).as_posix() if folder.is_relative_to(PROJECT_ROOT) else str(folder)


def assign_orientation(plans: list[dict]) -> None:
    """묶음(원재료, 한식 ...) 안에서 가로와 세로를 번갈아 줘서 전체와 묶음마다 반반이 되게 합니다."""
    seen: dict[str, int] = {}
    for plan in plans:
        group = plan["group"] or plan["category"]
        plan["orientation"] = ("landscape", "portrait")[seen.get(group, 0) % 2]
        seen[group] = seen.get(group, 0) + 1


def load_plans(run: Path, categories: list[str] | None = None) -> list[dict]:
    plans = json.loads((run / "plans.json").read_text(encoding="utf-8"))["plans"]
    return [p for p in plans if not categories or p["category"] in categories]


def jobs_of(plans: list[dict]) -> list[dict]:
    return [{"id": p["id"], "prompt": p["option"]["prompt"], "orientation": p["orientation"], "seed": p["seed"]}
            for p in plans]


def command_plan(photo_path: Path, run: Path) -> list[dict]:
    from .plan import Planner

    folder, photos = read_photos(photo_path)
    run.mkdir(parents=True, exist_ok=True)
    plans_file = run / "plans.json"
    done = {p["id"]: p for p in json.loads(plans_file.read_text(encoding="utf-8"))["plans"]} if plans_file.is_file() else {}
    planner, seed = Planner(), load_settings()["generation"]["seed"]

    def save() -> list[dict]:
        """해석이 끝난 사진까지 저장합니다. 중간에 끊겨도 다시 실행하면 남은 사진만 해석합니다."""
        plans = [done[p["id"]] | {"seed": seed + i} for i, p in enumerate(photos) if p["id"] in done]
        assign_orientation(plans)
        usage = planner.ai.usage
        plans_file.write_text(json.dumps({"photos_dir": portable(folder), "plans": plans,
                                          "usage": {"calls": len(usage),
                                                    "input_tokens": sum(u["input_tokens"] for u in usage),
                                                    "output_tokens": sum(u["output_tokens"] for u in usage)}},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
        return plans

    for photo in photos:
        if photo["id"] in done:
            continue
        plan = planner.plan(Image.open(folder / photo["file"]).convert("RGB"))
        # 비교 실험은 첫 번째 후보로 만듭니다. 나머지 후보는 사장님이 고르는 화면을 위해 남깁니다
        done[photo["id"]] = photo | plan | {"option": plan["options"][0]}
        print(f"[해석] {photo['id']}: {plan['category_label']} · {plan['name_ko']} → {plan['options'][0]['name_ko']}",
              flush=True)
        save()
    return save()


def calibration_plans(plans: list[dict], count: int) -> list[dict]:
    """성격이 다른 사진: 묶음을 돌아가며 하나씩 고릅니다."""
    groups: dict[str, list[dict]] = {}
    for plan in plans:
        groups.setdefault(plan["group"] or plan["category"], []).append(plan)
    picked, queues = [], list(groups.values())
    while len(picked) < count and any(queues):
        for queue in queues:
            if queue and len(picked) < count:
                picked.append(queue.pop(0))
    return picked


def main():
    parser = argparse.ArgumentParser(prog="python -m models.generative_ai.usage_examples", description="활용 예시 이미지 생성")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run", help="사진 한 장: 해석 → 생성 → 각도 → 업스케일 → 보정")
    one.add_argument("--input", type=Path, required=True)
    one.add_argument("--option", type=int, default=1, help="해석 모델이 낸 장면 후보 중 몇 번째로 만들지")
    one.add_argument("--orientation", choices=["landscape", "portrait"])
    one.add_argument("--preset", action="append", help="각도 프리셋 (기본: settings.toml)")
    step = commands.add_parser("plan", help="사진마다 해석 모델로 업종과 장면 후보 만들기")
    step.add_argument("--input", type=Path, required=True, help="사진 폴더 (sources.toml이 있으면 그 목록)")
    step.add_argument("--run", type=Path, required=True, help="실험 폴더")
    for name, text in (("generate", "장면을 조건별로 생성"), ("calibrate", "MagCache 크기 비율 재기")):
        step = commands.add_parser(name, help=text)
        step.add_argument("--run", type=Path, required=True)
        step.add_argument("--category", nargs="+", help="이 업종만 (기본: 전부)")
        step.add_argument("--condition", nargs="+", help="이 조건만 (기본: settings.toml 전부)")
        if name == "calibrate":
            step.add_argument("--count", type=int, default=6, help="비율을 잴 사진 수")
    step = commands.add_parser("finish", help="생성 이미지 한 장을 각도 → 업스케일 → 보정")
    step.add_argument("--run", type=Path, required=True)
    step.add_argument("--id", required=True)
    step.add_argument("--condition", default="original")
    step.add_argument("--preset", action="append")
    step = commands.add_parser("review", help="사람 평가 폴더 만들기")
    step.add_argument("--run", type=Path, required=True)
    step.add_argument("--out", type=Path, required=True)
    step.add_argument("--category", nargs="+", default=["food"])
    step = commands.add_parser("summarize", help="평가 CSV 합치기")
    step.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()

    if args.command == "plan":
        command_plan(args.input, args.run)
    elif args.command == "generate":
        from .generate import generate

        print(json.dumps(generate(jobs_of(load_plans(args.run, args.category)), args.run / "images", args.condition),
                         ensure_ascii=False, indent=2))
    elif args.command == "calibrate":
        from .generate import generate

        plans = calibration_plans(load_plans(args.run, args.category), args.count)
        generate(jobs_of(plans), args.run / "calibration", args.condition, calibrate=True)
        print(f"MagCache 비율을 저장했습니다 ({len(plans)}장)")
    elif args.command == "finish":
        from .finish import finish

        plan = next(p for p in load_plans(args.run) if p["id"] == args.id)
        print(finish(args.run / "images" / args.condition / f"{args.id}.png", args.run / "finish" / args.id,
                     plan["option"]["prompt"], plan["category"], args.preset))
    elif args.command == "review":
        from .review_page import build

        plans = load_plans(args.run, args.category)
        photos_dir = PROJECT_ROOT / json.loads((args.run / "plans.json").read_text(encoding="utf-8"))["photos_dir"]
        print(build(plans, args.run / "images", photos_dir, args.out))
    elif args.command == "summarize":
        from .review_page import summarize

        print(json.dumps(summarize(args.csv), ensure_ascii=False, indent=2))
    elif args.command == "run":
        from .finish import finish
        from .generate import generate

        run = project_path("output_dir") / args.input.stem
        plan = command_plan(args.input, run)[0]
        plan["option"] = plan["options"][min(args.option, len(plan["options"])) - 1]
        plan["orientation"] = args.orientation or load_settings()["generation"]["default_orientation"]
        condition = load_settings()["generation"]["default_condition"]
        generate(jobs_of([plan]), run / "images", [condition])
        print(finish(run / "images" / condition / f"{plan['id']}.png", run / "finish" / plan["id"],
                     plan["option"]["prompt"], plan["category"], args.preset))


if __name__ == "__main__":
    main()
