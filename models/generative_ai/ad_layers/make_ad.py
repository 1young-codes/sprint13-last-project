"""보정까지 끝난 제품 사진으로 레이어가 나뉜 광고 이미지를 만듭니다 (그래픽 카드 필요).

1. 제품을 잘라 캔버스에 놓기 (BiRefNet 마스크)               → product.png
2. 언어모델이 사장님 한 줄을 긴 장면 설명으로 풀기
3. 생성 모델이 제품 없는 배경을 그리기                       → background.png
4. 제품 마스크에서 바닥에 닿는 그림자를 만들기               → shadow.png
5. 제목·부제·가격 배치에 맞춰 문구를 준비 (가격은 입력값 그대로)
6. 고정 폰트로 글자 줄마다 그림                              → text_*.png
7. 합치기                                                    → ad.png, layout.json

    python -m models.generative_ai.ad_layers.make_ad 사진.png --name "딸기 생크림 케이크" --unit "1조각" --price 6500 \
        --usage "오후 커피와 함께" --mood all --idea "햇살 드는 카페 테이블"
    python -m models.generative_ai.ad_layers.make_ad --recompose outputs/ad_layers/사진/minimal   # 문구만 고친 뒤 다시 합치기
"""

import argparse
import time
from pathlib import Path

from PIL import Image

from .copywriter import scene_description, write_copy
from .layers import (PROJECT_ROOT, compose, contact_shadow, load_layout, load_settings, place_product,
                     save_layout, template_boxes, to_float)
from .text_render import render_texts


def load_models() -> dict:
    """한 번만 부릅니다: 생성 모델(Qwen-Image-2.1), 제품 마스크(BiRefNet), 언어모델(OpenAI)."""
    from ..fast_image_generation.qwen_image_model import QwenImageModel
    from ..photo_retouch.interpret import Interpreter
    from ..photo_retouch.product_mask import load_mask_model

    # ponytail: MagCache 비율 없이 모든 단계를 계산합니다. 속도가 필요하면 광고 사진으로 비율을 재서 넘깁니다.
    return {"generation": QwenImageModel(), "mask": load_mask_model(), "ai": Interpreter()}


def make_ad(photo: Image.Image, fields: dict, mood_key: str, idea: str, models: dict, folder: Path,
            seed: int | None = None, steps: int | None = None) -> dict:
    """folder에 레이어 파일과 layout.json, 합친 ad.png를 저장하고 layout을 돌려줍니다."""
    from ..photo_retouch.product_mask import product_mask

    settings = load_settings()
    generation, mood = settings["generation"], settings["moods"][mood_key]
    seed = generation["seed"] if seed is None else seed
    steps = steps or generation["steps"]
    model, ai = models["generation"], models["ai"]
    size = (settings["canvas"]["width"], settings["canvas"]["height"])
    folder.mkdir(parents=True, exist_ok=True)
    timings, started = {}, time.perf_counter()

    def lap(name):
        nonlocal started
        timings[name] = round(time.perf_counter() - started, 1)
        started = time.perf_counter()

    # 1. 제품 레이어
    mask = product_mask(models["mask"], to_float(photo))
    product, x, y = place_product(photo, mask, size, settings["product"])
    product.save(folder / "product.png")
    start = Image.new("RGB", size, settings["canvas"]["background_color"])
    start.paste(product, (x, y), product)
    lap("product")

    # 배경 모델에는 제품 사진 대신 빈 캔버스를 전달합니다.
    scene_text, view = scene_description(ai, to_float(start), fields["product_name"], idea, mood)
    view_prompt = generation[f"{view}_view"]
    prompt = generation["background_prompt"].format(scene=scene_text, view=view_prompt)
    model.use_lora(None, 0)
    blank = Image.new("RGB", size, settings["canvas"]["background_color"])
    background = model.edit(blank, prompt, seed, steps).resize(size, Image.Resampling.LANCZOS)
    background.save(folder / "background.png")
    lap("background")

    layout = {"size": list(size), "mood": mood_key, "seed": seed, "fields": fields, "scene_prompt": scene_text, "view": view,
              "background": {"file": "background.png"},
              "product": {"file": "product.png", "x": x, "y": y}, "texts": []}
    layout["shadow"] = save_shadow(folder, product, view)
    base = compose(layout, folder)
    base.save(folder / "base.png")
    lap("shadow")

    write_texts(folder, layout, ai, base)
    lap("text")
    layout["seconds"] = timings
    save_layout(layout, folder)
    return layout


def save_shadow(folder: Path, product: Image.Image, view: str) -> dict:
    """그림자를 제품 기준 상대 좌표로 저장합니다."""
    shadow, dx, dy = contact_shadow(product, view == "overhead")
    shadow.save(folder / "shadow.png")
    return {"file": "shadow.png", "dx": dx, "dy": dy}


def write_texts(folder: Path, layout: dict, ai, base: Image.Image) -> None:
    """제목, 부제, 가격을 기본 배치에 맞춰 별도 글자 레이어로 그립니다."""
    settings = load_settings()
    mood = settings["moods"][layout["mood"]]
    layout["text_source"] = "template"
    base_pixels = to_float(base)
    boxes = template_boxes(base_pixels, settings["template"])
    copy = write_copy(ai, base_pixels, boxes, layout["fields"], mood)
    limits = settings["text_layout"]["font_limits"]
    layout["texts"] = [box | line | {"font": mood["fonts"][line["role"]], "file": f"text_{i}_{line['role']}.png",
                                     "max_font_size": round(limits[line["role"]] * layout["size"][1])}
                       for i, (box, line) in enumerate(zip(boxes, copy), start=1) if line["role"] != "skip"]
    render_texts(layout["texts"], tuple(layout["size"]), folder)
    compose(layout, folder).save(folder / "ad.png")
    used = {line["file"] for line in layout["texts"]}
    for old in folder.glob("text_*.png"):
        if old.name not in used:
            old.unlink()


def relayout(folder: Path, ai) -> None:
    """저장된 배경과 제품 레이어로 그림자와 글자를 다시 계산합니다."""
    layout = load_layout(folder)
    product_info = layout["product"]
    with Image.open(folder / product_info["file"]) as product:
        layout["shadow"] = save_shadow(folder, product, layout.get("view", "angled"))
    base = compose(layout | {"texts": []}, folder)
    base.save(folder / "base.png")
    write_texts(folder, layout, ai, base)
    save_layout(layout, folder)


def recompose(folder: Path) -> None:
    """layout.json의 문구·폰트·색·위치를 고친 뒤 글자 레이어만 다시 그리고 합칩니다 (생성 모델 없이)."""
    layout = load_layout(folder)
    render_texts(layout["texts"], tuple(layout["size"]), folder)
    compose(layout, folder).save(folder / "ad.png")


def main() -> None:
    settings = load_settings()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("photo", nargs="?", type=Path, help="보정까지 끝난 제품 사진")
    p.add_argument("--recompose", type=Path, help="이 폴더의 layout.json으로 글자만 다시 그려 합치기")
    p.add_argument("--relayout", type=Path, help="이 폴더의 생성 결과로 그림자·글자 자리·문구를 다시 계산하기")
    p.add_argument("--name", help="품목 이름")
    p.add_argument("--unit", default="")
    p.add_argument("--price", type=int)
    p.add_argument("--usage", default="", help="활용 예시")
    p.add_argument("--notes", default="")
    p.add_argument("--idea", default="", help="사장님이 쓰는 장면 한 줄")
    p.add_argument("--mood", default="all", choices=[*settings["moods"], "all"])
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int)
    args = p.parse_args()
    if args.recompose:
        recompose(args.recompose)
        return
    if args.relayout:
        from ..photo_retouch.interpret import Interpreter

        relayout(args.relayout, Interpreter())
        return
    if not args.photo or not args.name or not args.price:
        p.error("사진, --name, --price가 필요합니다 (문구만 다시 합칠 때는 --recompose)")
    fields = {"product_name": args.name, "unit": args.unit, "price_krw": args.price,
              "usage_example": args.usage, "notes": args.notes}
    photo = Image.open(args.photo).convert("RGB")
    models = load_models()
    for mood in settings["moods"] if args.mood == "all" else [args.mood]:
        folder = PROJECT_ROOT / settings["paths"]["output_dir"] / args.photo.stem / mood
        layout = make_ad(photo, fields, mood, args.idea, models, folder, args.seed, args.steps)
        print(f"{mood}: {folder / 'ad.png'}  글자 {len(layout['texts'])}줄  {layout['seconds']}")


if __name__ == "__main__":
    main()
