import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import instagram_photo as ig  # noqa: E402


def test_upload_makes_jpeg_and_returns_signed_url(tmp_path, monkeypatch):
    src = tmp_path / "in.png"
    Image.new("RGBA", (64, 48), (10, 20, 30, 255)).save(src)
    calls = []

    def fake(*args):
        calls.append(args)
        if args[1] == "cp":  # 올라간 파일은 JPEG여야 한다
            assert Image.open(args[2]).format == "JPEG" and args[3].startswith(ig.BUCKET + "/")
            return ""
        return "expiration: x\nsigned_url: https://h.example/x.jpg?sig=1\n"

    monkeypatch.setattr(ig, "_gcloud", fake)
    assert ig.upload_for_instagram(src) == "https://h.example/x.jpg?sig=1"
    assert calls[1][0:2] == ("storage", "sign-url")


def test_delete_removes_only_our_bucket_object(monkeypatch):
    calls = []
    monkeypatch.setattr(ig, "_gcloud", lambda *a: calls.append(a) or "")
    ig.delete_from_instagram_bucket(f"https://{ig.BUCKET_HOST}/abc.jpg?x-goog-signature=1")
    assert calls == [("storage", "rm", f"{ig.BUCKET}/abc.jpg")]


def test_delete_refuses_other_host(monkeypatch):
    monkeypatch.setattr(ig, "_gcloud", lambda *a: pytest.fail("지우면 안 됩니다"))
    with pytest.raises(ValueError):
        ig.delete_from_instagram_bucket("https://example.com/abc.jpg")
