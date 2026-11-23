"""MagCache 크기 비율을 재서 mag_cache_ratios.json에 저장합니다 (그래픽 카드 필요).

    python -m backend.generative_ai.camera_angle_presets.calibrate_mag_cache --input 사진1.jpg --input 사진2.jpg --preset all

모든 단계를 계산하며 단계마다 변화량의 크기 비율을 재고, LoRA별로 모든 사진·프리셋의 평균을 냅니다.
특정 사진에 맞춰지지 않도록 성격이 다른 사진을 여러 장 쓰는 것이 좋습니다.
"""

import argparse
from pathlib import Path

from PIL import Image

from ..fast_image_generation.qwen_image_caching.mag_cache import save_ratios
from .change_camera_angle import RATIOS_FILE, change_camera_angle
from .settings import load_angle_presets, load_settings


def main():
    parser = argparse.ArgumentParser(prog="python -m backend.generative_ai.camera_angle_presets.calibrate_mag_cache",
                                     description="MagCache 크기 비율을 잽니다")
    parser.add_argument("--input", type=Path, action="append", required=True, help="사진 경로 (여러 번 지정 가능)")
    parser.add_argument("--preset", action="append", required=True, help="프리셋 이름 (여러 번 지정 가능, all이면 전부)")
    args = parser.parse_args()

    from ..fast_image_generation.qwen_image_model import QwenImageModel

    names = list(load_angle_presets()) if "all" in args.preset else args.preset
    model = QwenImageModel(ratios_file=RATIOS_FILE)
    model.calibration = {}
    for path in args.input:
        with Image.open(path) as photo:
            _, info = change_camera_angle(photo, names, model=model)
        print(f"{path.name}: 프리셋별 초 (MagCache 없이) {info['seconds_per_preset']}", flush=True)
    save_ratios(RATIOS_FILE, model.calibration, load_settings()["generation"]["steps"], len(args.input))
    print(f"저장했습니다: {RATIOS_FILE.name} ({', '.join(f'{k} {len(v)}회' for k, v in model.calibration.items())})")


if __name__ == "__main__":
    main()
