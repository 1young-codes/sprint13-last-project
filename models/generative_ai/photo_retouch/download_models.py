"""model_sources.toml의 모델을 허깅페이스 캐시(HF_HOME)에 받습니다. 이미 있으면 다시 받지 않습니다.

    python -m models.generative_ai.photo_retouch.download_models            # 전부
    python -m models.generative_ai.photo_retouch.download_models vlm        # 일부만
"""

import sys
from pathlib import Path

from .settings import read_toml


def model_folder(key: str, download: bool = False) -> Path:
    """모델 스냅샷 폴더. download=False면 캐시에 없을 때 안내와 함께 멈춥니다."""
    from huggingface_hub import snapshot_download

    entry = read_toml("model_sources.toml")["models"][key]
    try:
        return Path(snapshot_download(entry["repo"], revision=entry["revision"], allow_patterns=entry["files"],
                                      local_files_only=not download))
    except Exception as error:
        if download:
            raise
        raise FileNotFoundError(f"모델 {key}({entry['repo']})가 캐시에 없습니다. "
                                "python -m models.generative_ai.photo_retouch.download_models 를 먼저 실행하세요") from error


if __name__ == "__main__":
    for model_key in sys.argv[1:] or read_toml("model_sources.toml")["models"]:
        print(f"[받는 중] {model_key}: {model_folder(model_key, download=True)}")
