"""설정 파일(.toml)과 .env를 읽습니다. 코드에는 경로나 숫자를 직접 쓰지 않습니다."""

import os
import tomllib
from functools import cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
ENV_PREFIX = "RETOUCH_"


@cache
def read_toml(name: str) -> dict:
    with open(PACKAGE_DIR / name, "rb") as file:
        return tomllib.load(file)


def load_settings() -> dict:
    return read_toml("settings.toml")


def load_categories() -> dict:
    return read_toml("categories.toml")


def env_value(name: str, prefixed: bool = True) -> str | None:
    """환경 변수 RETOUCH_<NAME>(prefixed=False면 NAME 그대로), 없으면 프로젝트 루트 .env의 같은 키."""
    key = ENV_PREFIX + name.upper() if prefixed else name
    if key in os.environ:
        return os.environ[key]
    path = PROJECT_ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip() == key and v.strip():
                return v.strip().strip("'\"")
    return None


def project_path(name: str) -> Path:
    return PROJECT_ROOT / (env_value(name) or load_settings()["paths"][name])


def category_rules(category: str) -> dict:
    """업종 규칙에 settings.toml 기본값을 합친 것: bounds(값 범위), fidelity(검사 기준)."""
    categories = load_categories()
    if category not in categories:
        raise ValueError(f"업종 {category}가 없습니다 (가능: {', '.join(categories)})")
    rule = categories[category]
    base = load_settings()
    bounds = {k: list(v) for k, v in base["params"].items()}
    for key, (low, high) in rule.get("bounds", {}).items():
        bounds[key] = [max(low, bounds[key][0]), min(high, bounds[key][1])]
    return rule | {"name": category, "bounds": bounds, "fidelity": base["fidelity"] | rule.get("fidelity", {})}
