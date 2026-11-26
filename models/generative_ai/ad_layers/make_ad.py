"""보정까지 끝난 제품 사진으로 레이어가 나뉜 광고 이미지를 만듭니다 (그래픽 카드 필요).

1. 제품을 잘라 캔버스에 놓기 (BiRefNet 마스크)               → product.png
2. 언어모델이 사장님 한 줄을 긴 장면 설명으로 풀기
3. 생성 모델이 제품 둘레 장면을 그리기 (글자 없이)
4. 제품과 그 둘레를 지우고 빈 배경으로 채우기                → background.png
5. 장면과 빈 배경의 밝기 차이로 그림자·반사 분리              → shadow.png
6. 합친 그림 위에 생성 모델이 가짜 글자 초안을 그리고, 바뀐 자리에서 글자 줄 위치·크기·색을 읽음 (초안은 확인용으로만 남김)
7. 언어모델이 글자 자리마다 역할과 문구 초안을 씀 (가격은 입력값 그대로)
8. 고정 폰트로 글자 줄마다 그림                                → text_*.png
9. 합치기                                                       → ad.png, layout.json

    python -m models.generative_ai.ad_layers.make_ad 사진.png --name "딸기 생크림 케이크" --unit "1조각" --price 6500 \
        --usage "오후 커피와 함께" --mood all --idea "햇살 드는 카페 테이블"
    python -m models.generative_ai.ad_layers.make_ad --recompose outputs/ad_layers/사진/minimal   # 문구만 고친 뒤 다시 합치기
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .copywriter import scene_description, write_copy
from .layers import (PROJECT_ROOT, compose, find_text_boxes, load_layout, load_settings, place_product,
                     product_canvas_mask, save_layout, shadow_layer, shadow_region, to_float)
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
    from ..camera_angle_presets.fill_empty_area import fill_empty_area
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
    on_product = product_canvas_mask(product, x, y, size)
    start = Image.new("RGB", size, settings["canvas"]["background_color"])
    start.paste(product, (x, y), product)
    lap("product")

    # 2~3. 장면
    scene_text = scene_description(ai, to_float(start), fields["product_name"], idea, mood)
    scene = model.edit(start, generation["scene_prompt"].format(scene=scene_text), seed, steps).resize(size)
    scene.paste(product, (x, y), product)  # 제품은 원본 화소 그대로
    scene.save(folder / "scene.png")
    lap("scene")

    # 4. 빈 배경: 제품과 둘레를 표시색으로 칠해 채웁니다 (둘레 밖 화소는 장면 그대로)
    shadow_rule = settings["shadow"]
    region = shadow_region(on_product, shadow_rule["reach"], settings["product"]["mask_threshold"])
    region_mask = Image.fromarray((region * 255).astype(np.uint8))
    marked = Image.composite(Image.new("RGB", size, "#00ff00"), scene, region_mask)
    background = fill_empty_area(model, marked, region_mask, generation["empty_scene"].format(scene=scene_text),
                                 seed, steps)
    background.save(folder / "background.png")
    lap("background")

    # 5. 그림자·반사
    layout = {"size": list(size), "mood": mood_key, "seed": seed, "fields": fields, "scene_prompt": scene_text,
              "background": {"file": "background.png"},
              "product": {"file": "product.png", "x": x, "y": y}, "texts": []}
    layout["shadow"] = save_shadow(folder, layout, scene, background)
    base = compose(layout, folder)
    base.save(folder / "base.png")
    lap("shadow")

    # 6. 글자 초안 (초안과 base의 차이로 글자 자리를 찾습니다)
    model.edit(base, generation["draft_prompt"].format(style=mood["style"]), seed, steps).resize(size).save(
        folder / "draft.png")
    lap("draft")

    # 7~9. 글자 자리, 문구, 글자 레이어, 합치기
    write_texts(folder, layout, ai)
    lap("text")
    layout["seconds"] = timings
    save_layout(layout, folder)
    return layout


def save_shadow(folder: Path, layout: dict, scene: Image.Image, background: Image.Image) -> dict:
    """그림자·반사 레이어를 저장합니다. 제품 위치 기준 상대 좌표라 제품을 옮기면 같이 옮겨집니다."""
    settings = load_settings()
    product, x, y = Image.open(folder / layout["product"]["file"]), layout["product"]["x"], layout["product"]["y"]
    on_product = product_canvas_mask(product, x, y, scene.size)
    region = shadow_region(on_product, settings["shadow"]["reach"], settings["product"]["mask_threshold"])
    shadow = shadow_layer(to_float(scene), to_float(background), region, on_product, settings["shadow"])
    box = shadow.getbbox() or (x, y, x + 1, y + 1)
    shadow.crop(box).save(folder / "shadow.png")
    return {"file": "shadow.png", "dx": box[0] - x, "dy": box[1] - y}


def write_texts(folder: Path, layout: dict, ai) -> None:
    """draft.png와 base.png로 글자 자리를 찾고, 문구를 쓰고, 글자 레이어를 그려 ad.png로 합칩니다."""
    settings = load_settings()
    mood = settings["moods"][layout["mood"]]
    product = Image.open(folder / layout["product"]["file"])
    on_product = product_canvas_mask(product, layout["product"]["x"], layout["product"]["y"], tuple(layout["size"]))
    draft = to_float(Image.open(folder / "draft.png"))
    boxes = find_text_boxes(draft, to_float(Image.open(folder / "base.png")), on_product, settings["text_boxes"])
    copy = write_copy(ai, draft, boxes, layout["fields"], mood) if boxes else []
    for old in folder.glob("text_*.png"):
        old.unlink()
    layout["texts"] = [box | line | {"font": mood["fonts"][line["role"]], "file": f"text_{i}_{line['role']}.png"}
                       for i, (box, line) in enumerate(zip(boxes, copy), start=1) if line["role"] != "skip"]
    render_texts(layout["texts"], tuple(layout["size"]), folder)
    compose(layout, folder).save(folder / "ad.png")


def relayout(folder: Path, ai) -> None:
    """생성 모델 없이 그림자와 글자를 처음부터 다시 계산합니다 (설정값을 바꿔 볼 때).
    저장된 scene.png, background.png, base.png, draft.png를 그대로 씁니다."""
    layout = load_layout(folder)
    layout["shadow"] = save_shadow(folder, layout, Image.open(folder / "scene.png").convert("RGB"),
                                   Image.open(folder / "background.png").convert("RGB"))
    write_texts(folder, layout, ai)
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
