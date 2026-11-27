"""화면(React)이 부르는 게시글 만들기 API. 주소와 값은 docs/api-spec.md를 따른다.

켜고 끄는 방법은 serving/README.md.
"""
import argparse
import io
import logging
import os
import queue
import secrets
import sys
import threading
import time
import tomllib
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.generative_ai.llm.hashtags import generate_hashtags  # noqa: E402
from models.generative_ai.llm.product_captions import generate_product_captions  # noqa: E402

PRESETS_FILE = ROOT / "models/generative_ai/camera_angle_presets/angle_presets.toml"
OUTPUT_DIR = Path(os.environ.get("API_OUTPUT_DIR", "/data/outputs/jobs"))
MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 1024  # 원본 크기 그대로면 VM 메모리를 넘을 수 있다
FORMATS = {"JPEG", "PNG", "WEBP"}
ANGLE_FAILED = "이 각도는 만들지 못했습니다."
JOB_FAILED = "게시글을 만들지 못했습니다. 다시 시도해 주세요."
CAPTION_FAILED = "문구를 만들지 못했습니다. 다시 시도해 주세요."

log = logging.getLogger("api")


def read_presets() -> list[dict]:
    with open(PRESETS_FILE, "rb") as f:
        table = tomllib.load(f)
    return [{"id": k, "label": v["label"], "method": v["method"]} for k, v in table.items() if k != "original"]


PRESETS = read_presets()
PRESET_IDS = {p["id"] for p in PRESETS}

# ponytail: 작업 상태는 메모리에만 둔다. 서버를 다시 켜면 사라진다 (문서에 적힌 동작).
jobs: dict[str, dict] = {}
waiting: list[str] = []  # 순서를 기다리는 작업 번호
running = 0
lock = threading.Lock()
work: queue.Queue = queue.Queue()


def no_angle_model(photo: Image.Image, angle: str) -> Image.Image:
    raise RuntimeError("각도 모델을 불러오지 않았습니다 (--no-models)")


make_angle = no_angle_model  # 서버를 켤 때 모델을 불러 바꾼다. 테스트는 가짜로 바꾼다.
retouch_photo = None  # 서버를 켤 때 보정 모델을 불러 채운다. None이면 보정 없이 원본을 쓴다.


def load_retouch_model():
    """보정 모델은 한 번만 불러 두고, 사진마다 함수만 부른다. 업스케일은 빼고 1024 사진에 건다(11/27 회의 결정)."""
    from models.generative_ai.photo_retouch.retouch import load_models, retouch

    started = time.perf_counter()
    models = load_models()
    log.info("보정 모델 불러오기 %.1f초", time.perf_counter() - started)

    def run(photo):
        return retouch(photo, models)["image"]

    return run


def load_angle_model():
    """민기님 PR #13 방식: 모델은 한 번만 불러 두고, 각도마다 함수만 부른다."""
    from models.generative_ai.camera_angle_presets.change_camera_angle import change_camera_angle, load_models

    started = time.perf_counter()
    models = load_models(depth=True, generation=True)
    log.info("각도 모델 불러오기 %.1f초", time.perf_counter() - started)

    def angle(photo, name):
        # 한 장씩 불러야 고른 순서대로, 끝난 각도부터 화면에 보낼 수 있다
        results, _ = change_camera_angle(photo, [name], models)
        return results[0].image

    return angle


def make_texts(job_id: str, fields: dict) -> tuple[dict, dict, str | None]:
    """문구 두 편과 해시태그를 만든다. 문구가 실패하면 빈 문구와 이유를 돌려주고, 사진 결과는 그대로 둔다."""
    started = time.perf_counter()
    try:
        captions = generate_product_captions(**fields)
    except Exception:
        log.exception("%s 문구 실패", job_id)
        return {"product_caption": "", "usage_caption": ""}, {"product_photo": [], "usage_example": []}, CAPTION_FAILED
    log.info("%s 문구 %.1f초", job_id, time.perf_counter() - started)
    started = time.perf_counter()
    # 해시태그 함수는 실패해도 예외 없이 빈 목록을 돌려준다. 빈 목록이면 화면이 해시태그 줄을 숨긴다.
    tags = generate_hashtags(fields["product_name"], fields["usage_example"],
                             captions["product_caption"], captions["usage_caption"])
    log.info("%s 해시태그 %.1f초", job_id, time.perf_counter() - started)
    return captions, tags, None


def file_url(job_id: str, name: str) -> str:
    return f"/jobs/{job_id}/files/{name}"


def run_job(job_id: str) -> None:
    job = jobs[job_id]
    folder = OUTPUT_DIR / job_id
    photo = Image.open(folder / "input.png")
    post = job["product_photo"]
    shown = photo  # 대표 사진만 보정한다. 각도 사진은 보정 전 원본으로 만든다.
    if retouch_photo is not None:
        try:
            shown = retouch_photo(photo)
        except Exception:
            log.exception("%s 보정 실패, 원본을 씁니다", job_id)
    shown.save(folder / "product_photo.png")
    post["image_url"] = file_url(job_id, "product_photo.png")

    for angle in job["angles"]:
        started = time.perf_counter()
        try:
            name = f"angle_{angle}.png"
            make_angle(photo, angle).save(folder / name)
            item = {"angle": angle, "image_url": file_url(job_id, name), "error": None}
        except Exception:
            log.exception("%s 각도 %s 실패", job_id, angle)
            item = {"angle": angle, "image_url": None, "error": ANGLE_FAILED}
        log.info("%s 각도 %s %.1f초", job_id, angle, time.perf_counter() - started)
        with lock:
            post["angle_images"].append(item)

    captions, tags, error = make_texts(job_id, job["fields"])
    usage = {"image_url": None, "caption": captions["usage_caption"], "error": error,
             "hashtags": tags["usage_example"]}
    with lock:
        post["caption"], post["hashtags"], post["error"] = captions["product_caption"], tags["product_photo"], error
        job["usage_example"], job["status"] = usage, "ready"


def worker() -> None:
    global running
    while True:
        job_id = work.get()
        with lock:  # 대기열에서 빼기와 running 표시를 한 번에. 사이에 조회가 끼면 queue_position을 못 구한다
            waiting.remove(job_id)
            running += 1
            jobs[job_id].update(status="running", product_photo={
                "image_url": None, "angle_images": [], "caption": "", "hashtags": [], "error": None})
        started = time.perf_counter()
        try:
            run_job(job_id)
        except Exception:
            log.exception("%s 실패", job_id)
            with lock:
                jobs[job_id].update(status="failed", message=JOB_FAILED)
        finally:
            with lock:
                running -= 1
            log.info("%s 끝 %.1f초", job_id, time.perf_counter() - started)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=worker, daemon=True).start()
        yield

    app = FastAPI(title="게시글 만들기 API", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8504", "http://127.0.0.1:8504"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def korean_422(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "입력 형식이 올바르지 않습니다."})

    def bad(message: str):
        raise HTTPException(status_code=422, detail=message)

    @app.post("/jobs", status_code=202)
    def create_job(
        photo: UploadFile | None = File(None),
        product_name: str = Form(""),
        unit: str = Form(""),
        price_krw: str = Form(""),
        usage_example: str = Form(""),
        notes: str = Form(""),
        angles: list[str] = Form([]),
    ):
        if photo is None:
            bad("제품 사진을 올려 주세요.")
        # 화면에 그대로 보여주므로 값 이름 대신 화면에 보이는 이름으로 쓴다
        for label, value in (("품목 이름을", product_name), ("단위를", unit), ("활용 예시를", usage_example)):
            if not value.strip():
                bad(f"{label} 입력해 주세요.")
        price = price_krw.strip()
        if not price.isdecimal() or int(price) <= 0:
            bad("가격은 0보다 큰 숫자로 입력해 주세요.")
        unknown = [a for a in angles if a not in PRESET_IDS]
        if unknown:
            bad(f"없는 각도입니다: {', '.join(unknown)}")

        data = photo.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            bad("사진은 10MB 이하여야 합니다.")
        try:
            image = Image.open(io.BytesIO(data))
            if image.format not in FORMATS:
                bad("사진은 jpg, png, webp 파일이어야 합니다.")
            image = ImageOps.exif_transpose(image).convert("RGB")
        except HTTPException:
            raise
        except Exception:
            bad("사진 파일을 읽을 수 없습니다.")
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)  # 긴 변을 1024로. 작으면 그대로

        job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        folder = OUTPUT_DIR / job_id
        folder.mkdir(parents=True)
        image.save(folder / "input.png")
        fields = {"product_name": product_name.strip(), "unit": unit.strip(), "price_krw": int(price),
                  "usage_example": usage_example.strip(), "notes": notes.strip()}
        with lock:
            jobs[job_id] = {"job_id": job_id, "status": "queued", "message": "", "product_photo": None,
                            "usage_example": None, "fields": fields, "angles": list(angles)}
            waiting.append(job_id)
        work.put(job_id)
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="없는 작업 번호입니다.")
            post = job["product_photo"]
            return {
                "job_id": job_id,
                "status": job["status"],
                "message": job["message"],
                "queue_position": waiting.index(job_id) + running if job["status"] == "queued" else None,
                "product_photo": post and {**post, "angle_images": list(post["angle_images"]),
                                           "hashtags": list(post["hashtags"])},
                "usage_example": job["usage_example"],
            }

    @app.get("/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        path = OUTPUT_DIR / job_id / name
        if job_id not in jobs or name != Path(name).name or not name.endswith(".png") or not path.is_file():
            raise HTTPException(status_code=404, detail="없는 파일입니다.")
        return FileResponse(path, media_type="image/png")

    @app.get("/presets")
    def presets():
        return PRESETS

    return app


if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--no-models", action="store_true", help="각도 모델을 불러오지 않는다")
    p.add_argument("--port", type=int, default=8003)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        if not os.environ.get(key):
            log.warning("%s가 없습니다. README의 켜기 순서대로 키 파일을 불러와야 문구와 해시태그가 나옵니다.", key)
    if not args.no_models:
        # 요청을 받기 전에 불러온다. 실패하면 여기서 멈춰서, 모든 각도가 조용히 실패하는 서버가 뜨지 않는다.
        make_angle = load_angle_model()
        retouch_photo = load_retouch_model()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
