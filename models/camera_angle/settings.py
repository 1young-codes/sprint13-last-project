"""설정 파일(.toml)과 .env를 읽어 경로, 생성 값, 프리셋을 돌려줍니다.

코드에는 경로나 숫자를 직접 쓰지 않고, 모두 이 모듈을 거쳐 설정 파일에서 읽습니다.
"""

import os
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
METHODS = ("rotate", "orbit")


@dataclass(frozen=True)
class AnglePreset:
    name: str
    label: str
    method: str
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    fov_scale: float | None = None
    fov_deg: float | None = None
    lens_curve: float = 1.0
    aspect: float | None = None
    orbit_yaw_deg: float = 0.0
    orbit_pitch_deg: float = 0.0

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"프리셋 {self.name}: method는 {METHODS} 중 하나여야 합니다 (받은 값: {self.method})")
        if self.fov_scale is not None and self.fov_deg is not None:
            raise ValueError(f"프리셋 {self.name}: fov_scale과 fov_deg는 하나만 쓸 수 있습니다")
        if not 0 <= self.lens_curve <= 1:
            raise ValueError(f"프리셋 {self.name}: lens_curve는 0~1 사이여야 합니다")

    def output_fov_deg(self, photo_fov_deg: float, max_fov_deg: float) -> float:
        """이 프리셋으로 찍을 가로 화각. 일반 렌즈(lens_curve=1)만 max_fov_deg로 제한합니다."""
        if self.fov_deg is not None:
            fov = self.fov_deg
        else:
            fov = photo_fov_deg * (self.fov_scale or 1.0)
        return min(fov, max_fov_deg) if self.lens_curve == 1 else fov


def read_toml(name: str) -> dict:
    with open(PACKAGE_DIR / name, "rb") as file:
        return tomllib.load(file)


def read_env_file() -> dict[str, str]:
    """프로젝트 루트의 .env를 읽습니다. 이미 설정된 환경 변수가 우선합니다."""
    path = PROJECT_ROOT / ".env"
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values | {k: v for k, v in os.environ.items() if k.startswith("CAMERA_ANGLE_")}


@cache
def load_settings() -> dict:
    return read_toml("settings.toml")


def project_path(name: str) -> Path:
    """settings.toml [paths]의 name을 환경 변수 CAMERA_ANGLE_<NAME>으로 덮어쓴 뒤 프로젝트 루트 기준 경로로 바꿉니다."""
    value = read_env_file().get(f"CAMERA_ANGLE_{name.upper()}") or load_settings()["paths"][name]
    return PROJECT_ROOT / value  # value가 절대 경로면 그 경로가 그대로 쓰입니다


def load_angle_presets() -> dict[str, AnglePreset]:
    return {name: AnglePreset(name=name, **values) for name, values in read_toml("angle_presets.toml").items()}


def load_model_sources() -> dict:
    return read_toml("model_sources.toml")


def model_file(model_key: str) -> Path:
    """model_sources.toml에 적힌 모델의 첫 번째 파일 경로."""
    entry = load_model_sources()["models"][model_key]
    return project_path("model_dir") / entry["folder"] / entry["files"][0]


def model_folder(model_key: str) -> Path:
    return project_path("model_dir") / load_model_sources()["models"][model_key]["folder"]


def code_folder(code_key: str) -> Path:
    return project_path("model_dir") / load_model_sources()["code"][code_key]["folder"]
