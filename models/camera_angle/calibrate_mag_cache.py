"""MagCache 크기 비율을 재서 mag_cache_ratios.json에 저장합니다 (그래픽 카드 필요).

    python -m models.camera_angle.calibrate_mag_cache --input 사진1.jpg --input 사진2.jpg --preset all

모든 단계를 계산하며 단계마다 변화량의 크기 비율을 재고, LoRA별로 모든 사진·프리셋의 평균을 냅니다.
특정 사진에 맞춰지지 않도록 성격이 다른 사진을 여러 장 쓰는 것이 좋습니다.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .change_camera_angle import change_camera_angle
from .settings import load_angle_presets, load_settings
from .speed_up_generation import RATIOS_FILE


def main():
    parser = argparse.ArgumentParser(prog="python -m models.camera_angle.calibrate_mag_cache",
                                     description="MagCache 크기 비율을 잽니다")
    parser.add_argument("--input", type=Path, action="append", required=True, help="사진 경로 (여러 번 지정 가능)")
    parser.add_argument("--preset", action="append", required=True, help="프리셋 이름 (여러 번 지정 가능, all이면 전부)")
    args = parser.parse_args()

    from .qwen_image_model import QwenImageModel

    names = list(load_angle_presets()) if "all" in args.preset else args.preset
    model = QwenImageModel()
    model.calibration = {}
    for path in args.input:
        with Image.open(path) as photo:
            _, info = change_camera_angle(photo, names, model=model)
        print(f"{path.name}: 프리셋별 초 (MagCache 없이) {info['seconds_per_preset']}", flush=True)
    steps = load_settings()["generation"]["steps"]
    ratios = {name: [round(float(v), 5) for v in np.mean(runs, axis=0)] for name, runs in model.calibration.items()}
    record = {"steps": steps, "photos": len(args.input), "runs": {k: len(v) for k, v in model.calibration.items()},
              "ratios": ratios}
    RATIOS_FILE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"저장했습니다: {RATIOS_FILE.name} ({', '.join(f'{k} {len(v)}회' for k, v in model.calibration.items())})")


if __name__ == "__main__":
    main()
