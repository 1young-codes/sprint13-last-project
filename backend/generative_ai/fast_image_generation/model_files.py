"""model_sources.toml에 적힌 모델 파일을 허깅페이스 캐시(HF_HOME)에서 찾거나 내려받습니다.

    python -m backend.generative_ai.fast_image_generation.model_files    # 이 폴더의 Qwen-Image 모델 내려받기

기능마다 자기 폴더에 model_sources.toml을 두고, 이 모듈의 함수에 그 파일 경로를 넘겨 씁니다.
이미 캐시에 있는 파일(팀원이 받아 둔 것 포함)은 다시 받지 않습니다.
"""

import hashlib
import sys
import tomllib
from pathlib import Path

SOURCES_FILE = Path(__file__).resolve().parent / "model_sources.toml"


def read_sources(sources_file: Path = SOURCES_FILE) -> dict:
    with open(sources_file, "rb") as file:
        return tomllib.load(file)


def model_file(model_key: str, sources_file: Path = SOURCES_FILE) -> Path:
    """sources_file에 적힌 모델의 첫 번째 파일을 허깅페이스 캐시에서 찾습니다."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    entry = read_sources(sources_file)["models"][model_key]
    try:
        return Path(hf_hub_download(entry["repo"], entry["files"][0], revision=entry["revision"], local_files_only=True))
    except LocalEntryNotFoundError:
        raise FileNotFoundError(f"모델 {model_key}({entry['repo']})가 아직 없습니다. "
                                f"{sources_file.parent.name} 폴더 README의 모델 내려받기 명령을 먼저 실행하세요") from None


def model_folder(model_key: str, sources_file: Path = SOURCES_FILE) -> Path:
    """모델 저장소를 받아 둔 폴더 (설정 파일들이 저장소 안 경로 그대로 들어 있는 곳)."""
    first = read_sources(sources_file)["models"][model_key]["files"][0]
    return model_file(model_key, sources_file).parents[first.count("/")]


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(key: str, entry: dict) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    for filename in entry["files"]:
        try:
            hf_hub_download(entry["repo"], filename, revision=entry["revision"], local_files_only=True)
            print(f"[있음] {key}: {filename}")
            continue
        except LocalEntryNotFoundError:
            pass
        print(f"[받는 중] {key}: {entry['repo']}/{filename}")
        path = hf_hub_download(entry["repo"], filename, revision=entry["revision"])
        if entry.get("sha256") and sha256_of(path) != entry["sha256"]:
            raise ValueError(f"{key}: 내려받은 파일의 해시가 model_sources.toml과 다릅니다 ({path})")


if __name__ == "__main__":
    only = sys.argv[1:]
    for key, entry in read_sources()["models"].items():
        if not only or key in only:
            download_model(key, entry)
