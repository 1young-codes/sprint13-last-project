"""생성한 이미지 한 장을 기존 흐름으로 다듬습니다: 각도 → 업스케일 → 보정.

세 기능은 가상환경과 그래픽 메모리가 서로 달라서 각자의 명령으로 따로 실행하고, 결과 폴더만 이어 줍니다.
결과: <폴더>/1_angle/ (생성 원본 front.png + 각도 프리셋마다 한 장), 2_upscale/, 3_retouch/<이름>/retouched.png
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .settings import PROJECT_ROOT, load_settings, setting

MODULE = "models.generative_ai"


def python_for(stage: str) -> str:
    return setting("finish", f"{stage}_python") or sys.executable


def run_stage(command: list[str], env: dict | None = None) -> None:
    print("[실행]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ | (env or {}), check=True)


def finish(image: Path, folder: Path, scene: str, category: str, presets: list[str] | None = None) -> Path:
    """scene: 각도를 바꿀 때 드러난 빈 곳을 채울 장면 설명. category: 보정 업종. 보정 결과 폴더를 돌려줍니다."""
    settings = load_settings()["finish"]
    presets = presets or settings["angle_presets"]
    angle_dir, upscale_dir, retouch_dir = (folder / name for name in ("1_angle", "2_upscale", "3_retouch"))
    angle_dir.mkdir(parents=True, exist_ok=True)

    # 각도 기능은 결과를 <출력 폴더>/<입력 이름>/<프리셋>.png로 둡니다
    run_stage([python_for("angle"), "-m", f"{MODULE}.camera_angle_presets", "--input", str(image), "--scene", scene,
               *[arg for p in presets for arg in ("--preset", p)]],
              {"CAMERA_ANGLE_OUTPUT_DIR": str(angle_dir.resolve())})
    nested = angle_dir / image.stem
    for result in nested.iterdir():  # 그림과 함께 남는 run_info.json도 옮김
        shutil.move(result, angle_dir / result.name)
    nested.rmdir()
    shutil.copy(image, angle_dir / "front.png")  # 각도를 바꾸지 않은 생성 원본도 같이 다듬음

    run_stage([python_for("upscale"), "-m", f"{MODULE}.upscales.run_model", settings["upscale_model"],
               "--input", str(angle_dir), "--scale", str(settings["upscale_scale"]), "--output", str(upscale_dir)])
    run_stage([python_for("retouch"), "-m", f"{MODULE}.photo_retouch", *map(str, sorted(upscale_dir.glob("*.png"))),
               "--category", category],
              {"RETOUCH_OUTPUT_DIR": str(retouch_dir.resolve())})
    return retouch_dir
