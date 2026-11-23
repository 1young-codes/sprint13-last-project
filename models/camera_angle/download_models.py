"""model_sources.toml에 적힌 모델과 깊이 추정 코드를 settings.toml의 model_dir 아래로 내려받습니다.

    python -m models.camera_angle.download_models            # 전부
    python -m models.camera_angle.download_models moge2      # 일부만
"""

import hashlib
import io
import shutil
import sys
import tarfile
import urllib.request

from .settings import code_folder, load_model_sources, project_path


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(key: str, entry: dict) -> None:
    from huggingface_hub import hf_hub_download

    folder = project_path("model_dir") / entry["folder"]
    for filename in entry["files"]:
        target = folder / filename
        if target.is_file():
            print(f"[있음] {key}: {filename}")
            continue
        print(f"[받는 중] {key}: {entry['repo']}/{filename}")
        hf_hub_download(entry["repo"], filename, revision=entry["revision"], local_dir=folder)
        if entry.get("sha256") and sha256_of(target) != entry["sha256"]:
            target.unlink()
            raise ValueError(f"{key}: 내려받은 파일의 해시가 다릅니다. 파일을 지웠으니 다시 실행하세요")


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
    for key, entry in sources["models"].items():
        if not only or key in only:
            download_model(key, entry)
    for key, entry in sources["code"].items():
        if not only or key in only or (key == "moge" and {"moge2", "moge3"} & set(only)):
            download_code(key, entry)


if __name__ == "__main__":
    download_all_models(sys.argv[1:])
