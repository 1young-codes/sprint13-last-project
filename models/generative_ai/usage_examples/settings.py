"""설정 파일(.toml)과 .env를 읽습니다. 업종 표는 보정 기능의 categories.toml을 같이 씁니다."""

import tomllib
from functools import cache
from pathlib import Path

from ..photo_retouch.settings import env_value, load_categories  # noqa: F401  (업종 표는 한 곳에서 관리)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
RATIOS_FILE = PACKAGE_DIR / "mag_cache_ratios.json"


@cache
def read_toml(name: str) -> dict:
    with open(PACKAGE_DIR / name, "rb") as file:
        return tomllib.load(file)


def load_settings() -> dict:
    return read_toml("settings.toml")


def setting(section: str, name: str) -> str:
    """환경 변수 또는 .env의 USAGE_EXAMPLES_<NAME>, 없으면 settings.toml 값."""
    return env_value(f"USAGE_EXAMPLES_{name.upper()}", prefixed=False) or load_settings()[section][name]


def project_path(name: str) -> Path:
    return PROJECT_ROOT / setting("paths", name)  # 절대 경로면 그대로 쓰입니다


def conditions(names: list[str] | None = None) -> list[dict]:
    """settings.toml의 비교 조건. names가 있으면 그 순서대로 그 조건만."""
    table = {c["name"]: c for c in load_settings()["conditions"]}
    if names is None:
        return list(table.values())
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(f"없는 조건: {', '.join(unknown)} (가능: {', '.join(table)})")
    return [table[n] for n in names]


def condition_prompt(condition: dict, prompt: str) -> str:
    """로라 트리거 단어가 있으면 프롬프트 맨 앞에 붙입니다. 나머지는 세 조건이 같습니다."""
    trigger = condition.get("trigger", "")
    return f"{trigger}, {prompt}" if trigger else prompt
