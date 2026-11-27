"""인스타그램 게시용 사진을 임시 버킷에 올리고 서명된 주소를 만든다. 게시가 끝나면 지운다.

서버 VM의 gcloud로 한다. 서명은 서비스 계정 토큰 만들기 권한이 있어야 한다(11/26 설정).
"""
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

BUCKET = "gs://sprint-ai-chunk2-03-ig-temp"
BUCKET_HOST = "sprint-ai-chunk2-03-ig-temp.storage.googleapis.com"
SIGNER = "221528250454-compute@developer.gserviceaccount.com"
REGION = "us-central1"


def _gcloud(*args: str) -> str:
    return subprocess.run(["gcloud", *args], check=True, capture_output=True, text=True).stdout


def upload_for_instagram(photo_path: str | Path) -> str:
    """사진을 JPEG로 바꿔 올리고, 1시간 동안 열리는 주소를 돌려준다. 인스타그램은 JPEG만 받는다."""
    name = f"{uuid.uuid4().hex}.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        jpg = Path(tmp) / name
        Image.open(photo_path).convert("RGB").save(jpg, "JPEG", quality=95)
        _gcloud("storage", "cp", str(jpg), f"{BUCKET}/{name}")
    out = _gcloud("storage", "sign-url", f"{BUCKET}/{name}", "--duration=1h",
                  f"--impersonate-service-account={SIGNER}", f"--region={REGION}")
    for line in out.splitlines():
        if line.startswith("signed_url:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("서명된 주소를 찾지 못했습니다")


def delete_from_instagram_bucket(url: str) -> None:
    """upload_for_instagram이 돌려준 주소의 사진을 버킷에서 지운다. 다른 주소는 거부한다."""
    parsed = urlparse(url)
    if parsed.hostname != BUCKET_HOST:
        raise ValueError("이 버킷의 주소가 아닙니다")
    _gcloud("storage", "rm", f"{BUCKET}/{parsed.path.rsplit('/', 1)[-1]}")
