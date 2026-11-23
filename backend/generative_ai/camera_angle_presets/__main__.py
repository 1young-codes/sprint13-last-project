"""명령줄 실행.

    python -m models.camera_angle --list-presets
    python -m models.camera_angle --input 사진.jpg --preset dutch_angle --preset low_angle
    python -m models.camera_angle --input 사진.jpg --preset all --scene "a fishing boat on a calm sea"
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from .change_camera_angle import change_camera_angle
from .settings import PROJECT_ROOT, load_angle_presets, project_path


def main():
    parser = argparse.ArgumentParser(prog="python -m models.camera_angle", description="사진의 카메라 앵글을 프리셋대로 바꿉니다")
    parser.add_argument("--input", type=Path, help="입력 사진 경로")
    parser.add_argument("--preset", action="append", default=[], help="프리셋 이름 (여러 번 지정 가능, all이면 전부)")
    parser.add_argument("--list-presets", action="store_true", help="사용할 수 있는 프리셋을 보여 줍니다")
    parser.add_argument("--scene", default="", help="빈 곳을 채울 때 쓸 장면 설명 (영어 권장)")
    parser.add_argument("--seed", type=int, help="생성 시드 (기본값은 settings.toml)")
    parser.add_argument("--steps", type=int, help="생성 단계 수 (기본값은 settings.toml)")
    parser.add_argument("--photo-fov", type=float, help="원본 사진의 가로 화각(도). 주면 깊이 추정을 건너뜁니다")
    parser.add_argument("--skip-generation", action="store_true", help="생성 모델 없이 계산 결과만 확인합니다")
    parser.add_argument("--save-steps", action="store_true", help="중간 단계 이미지도 저장합니다")
    args = parser.parse_args()

    presets = load_angle_presets()
    if args.list_presets:
        for name, preset in presets.items():
            print(f"{name:22} {preset.method:7} {preset.label}")
        return
    if args.input is None or not args.preset:
        parser.error("--input과 --preset이 필요합니다 (프리셋 목록: --list-presets)")
    names = list(presets) if "all" in args.preset else args.preset

    with Image.open(args.input) as photo:
        results, info = change_camera_angle(photo, names, args.scene, args.seed, args.steps, args.photo_fov,
                                            args.skip_generation)
    folder = project_path("output_dir") / args.input.stem
    folder.mkdir(parents=True, exist_ok=True)
    for result in results:
        result.image.save(folder / f"{result.preset.name}.png")
        if args.save_steps:
            for step, image in result.steps.items():
                image.save(folder / f"{result.preset.name}_{step}.png")
    (folder / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = folder.relative_to(PROJECT_ROOT) if folder.is_relative_to(PROJECT_ROOT) else folder
    print(f"{len(results)}장을 저장했습니다: {shown}")


if __name__ == "__main__":
    main()
