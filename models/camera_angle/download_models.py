"""model_sources.toml에 적힌 모델을 허깅페이스 캐시(HF_HOME)로, 깊이 추정 코드를 settings.toml의 code_dir로 내려받습니다.

    python -m models.camera_angle.download_models            # 필요한 것 전부 (깊이 모델은 settings.toml에서 고른 것만)
    python -m models.camera_angle.download_models moge3      # 일부만

이미 캐시에 있는 파일(팀원이 받아 둔 것 포함)은 다시 받지 않습니다.
"""

import hashlib
import io
import shutil
import sys
import tarfile
import urllib.request

from .settings import code_folder, load_model_sources, load_settings

DEPTH_MODELS = ("moge2", "moge3")


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


def download_code(key: str, entry: dict) -> None:
    target = code_folder(key)
    if target.is_dir():
        print(f"[있음] 코드 {key}")
        return
    url = entry["archive_url"].format(revision=entry["revision"])
    print(f"[받는 중] 코드 {key}: {url}")
    with urllib.request.urlopen(url) as response:
        data = response.read()
    staging = target.with_name(target.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(staging, filter="data")
    (top,) = staging.iterdir()  # 압축 파일 안의 최상위 폴더 하나
    top.rename(target)
    staging.rmdir()


def download_all_models(only: list[str]) -> None:
    sources = load_model_sources()
    chosen_depth = load_settings()["depth"]["model"]
    for key, entry in sources["models"].items():
        wanted = key in only if only else (key not in DEPTH_MODELS or key == chosen_depth)
        if wanted:
            download_model(key, entry)
    for key, entry in sources["code"].items():
        if not only or key in only or set(DEPTH_MODELS) & set(only):
            download_code(key, entry)


if __name__ == "__main__":
    download_all_models(sys.argv[1:])
