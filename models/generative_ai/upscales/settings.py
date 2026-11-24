"""업스케일 공용 설정을 읽습니다.

코드에는 경로나 숫자를 직접 쓰지 않고, settings.toml과 모델 폴더의 setup.toml에서 읽습니다.
경로는 프로젝트 루트 기준이며 .env 또는 환경 변수 UPSCALE_<이름>으로 덮어씁니다.
"""

import os
import tomllib
from functools import cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
ENV_PREFIX = "UPSCALE_"
SPEC_FILE = "setup.toml"


def read_toml(path: Path) -> dict:
    with open(path, "rb") as file:
        return tomllib.load(file)


@cache
def load_settings() -> dict:
    return read_toml(PACKAGE_DIR / "settings.toml")


@cache
def env_values() -> dict[str, str]:
    """프로젝트 루트 .env의 UPSCALE_ 값. 이미 설정된 환경 변수가 우선합니다."""
    values = {}
    path = PROJECT_ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip().startswith(ENV_PREFIX):
                values[key.strip()] = value.strip().strip("'\"")
    return values | {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX)}


def project_path(name: str) -> Path:
    """settings.toml [paths]의 값. 상대 경로는 프로젝트 루트 기준입니다."""
    value = env_values().get(ENV_PREFIX + name.upper()) or load_settings()["paths"][name]
    return PROJECT_ROOT / value


def load_spec(name: str) -> dict:
    """모델 폴더의 setup.toml."""
    path = PACKAGE_DIR / name / SPEC_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path.relative_to(PROJECT_ROOT)}이 없습니다 (모델 목록: {', '.join(model_names())})")
    return read_toml(path)


def model_names() -> list[str]:
    """setup.toml에 [run]이 있는 폴더 = 업스케일 모델."""
    return sorted(p.parent.name for p in PACKAGE_DIR.glob(f"*/{SPEC_FILE}") if "run" in read_toml(p))


def code_folder(name: str) -> Path:
    return project_path("code_dir") / name


def venv_python(name: str) -> Path:
    return project_path("venv_dir") / name / "bin" / "python"
