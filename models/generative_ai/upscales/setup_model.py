"""모델별 가상환경, 추론 코드, 가중치를 준비합니다.

    python -m models.generative_ai.upscales.setup_model              # 실행 환경(runner)과 모든 모델
    python -m models.generative_ai.upscales.setup_model pixrestore   # 하나만

각 폴더의 setup.toml을 따릅니다.
- [environment]: 파이썬 버전과 requirements.txt로 settings.toml의 venv_dir 아래에 가상환경을 만듭니다.
- [code]: 깃허브 저장소를 고정 커밋으로 받아 code_dir 아래에 풉니다.
- [weights.*]: 허깅페이스 캐시(HF_HOME)에 고정 커밋으로 받습니다. 이미 있으면 다시 받지 않습니다.
- [links]: 추론 코드가 가중치를 찾는 자리에 캐시 파일을 링크로 연결합니다.
이미 끝난 단계는 건너뛰므로 여러 번 실행해도 됩니다.
"""

import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from .settings import PACKAGE_DIR, code_folder, load_settings, load_spec, model_names, venv_python

ARCHIVE_URL = "https://codeload.github.com/{repo}/tar.gz/{revision}"


def create_env(name: str, environment: dict) -> None:
    uv = load_settings()["tools"]["uv"]
    python = venv_python(name)
    if not python.exists():
        print(f"[가상환경] {name}: 파이썬 {environment['python']}")
        subprocess.run([uv, "venv", "--python", environment["python"], str(python.parents[1])], check=True)
    requirements = str(PACKAGE_DIR / name / "requirements.txt")
    installs = [["-r", requirements, *environment.get("pip_args", [])]]
    # 앞 설치가 끝나야 되는 패키지 (예: flash_attn). requirements.txt를 제약으로 걸어 고정 버전을 바꾸지 못하게 합니다
    installs += [[*args, "-c", requirements] for args in environment.get("extra_installs", [])]
    for args in installs:
        subprocess.run([uv, "pip", "install", "--python", str(python), *args], check=True)


def download_code(name: str, code: dict) -> None:
    target = code_folder(name)
    marker = target / ".revision"
    if marker.is_file() and marker.read_text().strip() == code["revision"]:
        print(f"[있음] 코드 {name}")
        return
    url = ARCHIVE_URL.format(**code)
    print(f"[받는 중] 코드 {name}: {url}")
    with urllib.request.urlopen(url) as response:
        data = response.read()
    staging = target.with_name(target.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(staging, filter="data")
    (top,) = staging.iterdir()  # 압축 파일 안의 최상위 폴더 하나
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    top.rename(target)
    staging.rmdir()
    marker.write_text(code["revision"])


def download_weights(weights: dict) -> dict:
    """가중치 키 -> 캐시 안 스냅샷 폴더."""
    from huggingface_hub import snapshot_download

    folders = {}
    for key, entry in weights.items():
        print(f"[가중치] {key}: {entry['repo']}")
        folders[key] = snapshot_download(entry["repo"], revision=entry["revision"], allow_patterns=entry["files"])
    return folders


def link_weights(name: str, links: dict, folders: dict) -> None:
    code = code_folder(name)
    for place, source in links.items():
        key, _, inner = source.partition(":")
        target = Path(folders[key]) / inner
        if not target.exists():
            raise FileNotFoundError(f"{name}: 링크 대상 {source}가 캐시에 없습니다. setup.toml의 files를 확인하세요")
        link = code / place
        if link.is_symlink():
            link.unlink()
        elif link.is_dir() and not any(p for p in link.iterdir() if not p.name.startswith(".")):
            shutil.rmtree(link)  # 저장소가 자리만 잡아 둔 빈 폴더
        elif link.exists():
            raise FileExistsError(f"{name}: {place}에 이미 파일이 있습니다. 덮어쓰지 않습니다")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)


def setup(name: str) -> None:
    spec = load_spec(name)
    if "code" in spec:
        download_code(name, spec["code"])
    folders = download_weights(spec.get("weights", {}))
    link_weights(name, spec.get("links", {}), folders)
    create_env(name, spec["environment"])
    print(f"[완료] {name}")


if __name__ == "__main__":
    for model in sys.argv[1:] or ["runner", *model_names()]:
        setup(model)
