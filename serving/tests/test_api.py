import io
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api  # noqa: E402

FIELDS = {"product_name": "참도미", "unit": "1마리", "price_krw": "25000", "usage_example": "회, 구이"}


def png(w=1600, h=900, fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 80, 60)).save(buf, fmt)
    return buf.getvalue()


def fake_angle(photo, angle):
    if angle == "wide_angle":
        raise RuntimeError("가짜 실패")
    return photo.rotate(10)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    api.OUTPUT_DIR = tmp_path_factory.mktemp("jobs")
    api.make_angle = fake_angle
    with TestClient(api.create_app()) as c:
        yield c


def post(client, data=FIELDS, photo=None, angles=()):
    files = {"photo": ("p.png", photo or png(), "image/png")}
    return client.post("/jobs", data={**data, "angles": list(angles)}, files=files)


def wait_done(client, job_id):
    for _ in range(100):
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("ready", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("작업이 끝나지 않았습니다")


def test_presets_skip_original(client):
    ids = [p["id"] for p in client.get("/presets").json()]
    assert len(ids) == 13 and "original" not in ids


def test_job_keeps_order_and_one_angle_failure(client):
    r = post(client, angles=["low_angle", "wide_angle", "tilt_up"])
    assert r.status_code == 202
    body = wait_done(client, r.json()["job_id"])
    assert body["status"] == "ready" and body["queue_position"] is None
    shots = body["product_photo"]["angle_images"]
    assert [s["angle"] for s in shots] == ["low_angle", "wide_angle", "tilt_up"]
    assert shots[1] == {"angle": "wide_angle", "image_url": None, "error": api.ANGLE_FAILED}
    assert shots[0]["error"] is None and body["usage_example"]["caption"]

    main = client.get(body["product_photo"]["image_url"])
    assert main.status_code == 200
    assert max(Image.open(io.BytesIO(main.content)).size) == 1024  # 1600x900 -> 긴 변 1024
    assert client.get(shots[0]["image_url"]).status_code == 200


def test_no_angles_gives_empty_list(client):
    body = wait_done(client, post(client).json()["job_id"])
    assert body["product_photo"]["angle_images"] == [] and body["product_photo"]["hashtags"] == []


@pytest.mark.parametrize("data, photo, angles, detail", [
    ({**FIELDS, "price_krw": "0"}, None, (), "가격은 0보다 큰 숫자로 입력해 주세요."),
    ({**FIELDS, "price_krw": "abc"}, None, (), "가격은 0보다 큰 숫자로 입력해 주세요."),
    ({**FIELDS, "unit": " "}, None, (), "단위를 입력해 주세요."),
    ({**FIELDS, "product_name": ""}, None, (), "품목 이름을 입력해 주세요."),
    ({**FIELDS, "usage_example": " "}, None, (), "활용 예시를 입력해 주세요."),
    (FIELDS, None, ("no_such",), "없는 각도입니다: no_such"),
    (FIELDS, b"not an image", (), "사진 파일을 읽을 수 없습니다."),
    (FIELDS, png(fmt="GIF"), (), "사진은 jpg, png, webp 파일이어야 합니다."),
    (FIELDS, b"0" * (api.MAX_BYTES + 1), (), "사진은 10MB 이하여야 합니다."),
], ids=["price-zero", "price-text", "unit-blank", "name-blank", "usage-blank", "angle-unknown", "not-image", "gif", "too-big"])
def test_bad_input_is_one_korean_sentence(client, data, photo, angles, detail):
    r = post(client, data=data, photo=photo, angles=angles)
    assert r.status_code == 422 and r.json() == {"detail": detail}


def test_missing_photo(client):
    r = client.post("/jobs", data=FIELDS)
    assert r.status_code == 422 and r.json() == {"detail": "제품 사진을 올려 주세요."}


def test_unknown_job_and_file(client):
    assert client.get("/jobs/nope").status_code == 404
    job_id = wait_done(client, post(client).json()["job_id"])["job_id"]
    assert client.get(f"/jobs/{job_id}/files/input.txt").status_code == 404
    assert client.get(f"/jobs/{job_id}/files/..%2F..%2Fsecret.png").status_code == 404


def test_cors_for_screen_dev_server(client):
    r = client.options("/presets", headers={"Origin": "http://localhost:8504", "Access-Control-Request-Method": "GET"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:8504"
