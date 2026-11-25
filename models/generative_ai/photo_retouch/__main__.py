"""사진을 자동 보정하고, 무엇을 왜 바꿨는지 한국어로 남깁니다.

    python -m models.generative_ai.photo_retouch 사진.jpg [사진2.png ...] [--category food] [--studio-light]

결과: <output_dir>/<사진 이름>/ 에 retouched.png, (선택) studio.png 또는 studio_rejected.png, compare.png, report.json, 설명.md
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from .retouch import load_models, retouch, unload_models
from .settings import load_categories, project_path

NAMES = {"exposure_ev": "밝기", "temperature": "색온도", "tint": "색조", "contrast": "대비", "highlights": "밝은 부분",
         "shadows": "어두운 부분", "vibrance": "생동감", "saturation": "채도"}
REGION_NAMES = {"global": "사진 전체", "product": "상품", "background": "배경"}


def explain(report: dict) -> str:
    lines = ["# 자동 보정 결과", "", f"- 업종: {report['category_label']}, 주인공: {report['subject']}"]
    lines += ["", "## 바꾼 것"]
    if report["chosen"] == "original":
        lines.append(f"- {report['chosen_reason']}")
    for region, params in report["params"].items():
        for key, value in params.items():
            if abs(value) >= 0.01:
                why = report["reasons"].get(region, {}).get(key, "")
                lines.append(f"- {REGION_NAMES[region]} {NAMES.get(key, key)} {value:+.2f}" + (f": {why}" if why else ""))
    if report.get("chosen_reason") and report["chosen"] != "original":
        lines.append(f"- 이 후보를 고른 이유: {report['chosen_reason']}")
    if report.get("removed"):
        lines.append(f"- 지운 것: {', '.join(report['removed'])}")
    if report.get("kept"):
        lines += ["", "## 일부러 그대로 둔 것", *[f"- {why}" for why in report["kept"].values()]]
    lines += ["", "## 실물 검사 (상품 영역)"]
    for candidate in report["candidates"]:
        f = candidate["fidelity"]
        mark = "통과" if f["ok"] else "탈락 - " + ", ".join(f["problems"])
        lines.append(f"- {candidate['name']}: 색 차이 {f['delta_e']:.1f}, 구조 {f['structure']:.3f} → {mark}")
    if "studio_light" in report:
        s = report["studio_light"]
        lines.append(f"- 스튜디오 조명: {'통과 (studio.png)' if s['ok'] else '쓰지 않음 - ' + ', '.join(s['problems'])}")
    return "\n".join(lines) + "\n"


def side_by_side(*pictures: Image.Image) -> Image.Image:
    width, height = pictures[0].size
    sheet = Image.new("RGB", (width * len(pictures), height))
    for index, picture in enumerate(pictures):
        sheet.paste(picture, (index * width, 0))
    return sheet


def main():
    parser = argparse.ArgumentParser(description="사진 자동 보정")
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--category", choices=list(load_categories()), help="없으면 해석 모델이 판단")
    parser.add_argument("--no-ai", action="store_true", help="사진 해석(OpenAI) 없이 기본 자동 보정만")
    # ponytail: 해석 모델이 주는 물건 좌표가 부정확해 기본은 끔. 정확한 위치 찾기 모델을 붙이면 기본으로 켤 것
    parser.add_argument("--cleanup", action="store_true", help="해석 모델이 짚은 배경 잡동사니를 지움 (위치가 틀릴 수 있음)")
    parser.add_argument("--studio-light", action="store_true", help="생성형 스튜디오 조명 결과도 만듦")
    args = parser.parse_args()

    models = load_models(use_ai=not args.no_ai, cleanup=args.cleanup, studio_light=args.studio_light)
    for photo_path in args.photos:
        photo = Image.open(photo_path).convert("RGB")
        output = retouch(photo, models, args.category)
        folder = project_path("output_dir") / photo_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        output["image"].save(folder / "retouched.png")
        pictures = [photo, output["image"]]
        if "studio_image" in output:
            output["studio_image"].save(folder / "studio.png")
            pictures.append(output["studio_image"])
        if "studio_rejected" in output:  # 검사에서 떨어진 결과도 확인용으로 남김
            output["studio_rejected"].save(folder / "studio_rejected.png")
        side_by_side(*pictures).save(folder / "compare.png")
        (folder / "report.json").write_text(json.dumps(output["report"], ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "설명.md").write_text(explain(output["report"]), encoding="utf-8")
        print(f"[완료] {photo_path.name} → {folder}")
    unload_models(models)


if __name__ == "__main__":
    main()
