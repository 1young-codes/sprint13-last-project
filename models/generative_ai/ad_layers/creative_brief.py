"""상품 사진과 입력 정보에서 광고 디자인을 기획하고 생성 지시를 저장합니다."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from ..photo_retouch.interpret import Interpreter, parse_json
from .layers import to_float

MODEL = "gpt-6-luna"
LAYOUT_ROLES = ("product", "headline", "product_name", "price")

PLANNING = """You handle the ad strategy. Decide on one message for this product.
Use the photo, supplied facts and owner's idea to choose one appeal the ad can demonstrate visually.
Return JSON with "message" (one Korean sentence about what viewers should take away) and "evidence"
(a Korean sentence identifying the exact supplied or visible fact supporting it). Do not write ad copy yet."""

DIRECTION = """You handle photographic direction. Keep the supplied strategy and express it in one coherent scene.
Use the photo to identify the product features to retain and emphasize. Choose the setting, camera viewpoint,
lighting direction, palette and product detail that makes the agreed message visible. Change the background
when it serves that message. Describe the intended scene, leaving placement decisions to the layout stage.
Return only JSON with one English string, "scene". Do not write ad copy."""

COPY = """You write the headline. Use the agreed message and scene to write one short, natural Korean line that
fits this particular product. Keep all factual claims grounded in the supplied information. The product name
and price have their own lines. Keep the headline free of prices. Return only JSON with "headline"."""

LAYOUT = """You handle layout. Use the supplied scene and exact copy. Decide where and at what scale the product,
headline, product name and price appear. Return one JSON object with:
- "boxes": one [x, y, width, height] array each for "product", "headline", "product_name" and "price".
  Use whole percentages of the canvas from its top-left corner. Each box must fit entirely inside the canvas
  and have positive width and height. Leave room to read all three strings without obscuring the product.
- "reading_order": an array listing the four roles from first seen to last, each exactly once.
- "typography": one English sentence describing type weight, alignment and contrast.
Keep the chosen scene and supplied wording. Include design directions only."""


def ask_stage(ai, step, task, inputs, images=()):
    request = (
        f"{task}\nUse the supplied inputs as data. Do not invent brands, offers, ingredients, "
        "certifications or health benefits.\nInput JSON follows:\n"
        + json.dumps(inputs, ensure_ascii=False)
    )
    return parse_json(ai.ask(step, list(images), request, model=MODEL))


def required_text(answer, key, step):
    value = answer.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{step} 단계에서 {key} 항목이 빠졌습니다")
    return value.strip()


def layout_boxes(answer):
    boxes = answer.get("boxes")
    if not isinstance(boxes, dict) or boxes.keys() != set(LAYOUT_ROLES):
        raise ValueError("레이아웃에 제품, 제목, 상품명, 가격 위치가 모두 필요합니다")
    for role, coordinates in boxes.items():
        if (not isinstance(coordinates, list) or len(coordinates) != 4
                or any(type(value) is not int for value in coordinates)):
            raise ValueError(f"레이아웃의 {role} 위치는 정수 네 개여야 합니다")
        x, y, width, height = coordinates
        if not (0 <= x < x + width <= 100 and 0 <= y < y + height <= 100):
            raise ValueError(f"레이아웃의 {role} 위치가 캔버스 범위를 벗어났습니다")
    order = answer.get("reading_order")
    if not isinstance(order, list) or len(order) != len(LAYOUT_ROLES) or any(order.count(role) != 1 for role in LAYOUT_ROLES):
        raise ValueError("레이아웃의 읽는 순서에 제품, 제목, 상품명, 가격이 각각 한 번씩 필요합니다")
    return {"boxes": boxes, "reading_order": order, "typography": required_text(answer, "typography", "레이아웃")}


def brainstorm(ai, photo, fields, idea, price):
    inputs = {key: fields[key] for key in ("product_name", "unit", "usage_example", "notes") if key in fields}
    photo_inputs = [to_float(photo)]
    strategy = ask_stage(ai, "ad_plan", PLANNING, inputs | {"owner_idea": idea}, photo_inputs)
    plan = {key: required_text(strategy, key, "기획") for key in ("message", "evidence")}

    direction = ask_stage(ai, "ad_direction", DIRECTION, inputs | {"idea": idea, "plan": plan}, photo_inputs)
    scene = required_text(direction, "scene", "연출")

    writing = ask_stage(ai, "ad_headline", COPY, inputs | {"plan": plan, "scene": scene})
    headline = " ".join(required_text(writing, "headline", "문구").split())

    copy = {"headline": headline, "product_name": fields["product_name"], "price": price}
    layout = ask_stage(ai, "ad_layout", LAYOUT, {"scene": scene, "message": plan["message"], "copy": copy})
    return plan | {"scene": scene, "headline": headline} | layout_boxes(layout)


def compile_prompt(brief, title, price):
    copy = json.dumps({"headline": brief["headline"], "product_name": title, "price": price}, ensure_ascii=False)
    return (
        "Create a finished Korean advertisement using the product in the reference photo. "
        "Keep the product recognizable. Execute this campaign direction:\n"
        f"Message: {brief['message']}\nScene: {brief['scene']}\n"
        "Layout boxes as [left, top, width, height], in percentages of the canvas: "
        f"{json.dumps(brief['boxes'])}\n"
        f"Reading order: {', '.join(brief['reading_order'])}\nTypography: {brief['typography']}\n"
        f"The complete advertising copy is given here. Render all three strings verbatim: {copy}. "
        "Do not add advertising copy or logos. Keep the text inside the poster and clearly legible."
    )


def prepare_ads(ai, items, photos):
    ads = []
    for index, item in enumerate(items, 1):
        reference = item["file"]
        fields = item["fields"]
        title, price = fields["product_name"], f"{fields['price_krw']:,}원"
        with Image.open(photos / reference) as opened:
            photo = ImageOps.exif_transpose(opened).convert("RGB")
        try:
            brief = brainstorm(ai, photo, fields, item.get("idea", ""), price)
        except ValueError as error:
            raise ValueError(f"{reference} · {error}") from error
        ads.append({
            "file": f"{index:02d}_{Path(reference).stem}.png",
            "reference": reference, "title": title, "price": price,
            "brief": brief, "prompt": compile_prompt(brief, title, price),
        })
    return {"design_model": MODEL, "ads": ads}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items", type=Path, help="파일명, 장면 한 줄, 상품 정보를 담은 JSON")
    parser.add_argument("photos", type=Path, help="참고 사진 폴더")
    parser.add_argument("output", type=Path, help="광고 디자인과 생성 지시를 저장할 JSON")
    args = parser.parse_args()
    items = json.loads(args.items.read_text(encoding="utf-8"))
    plan = prepare_ads(Interpreter(), items, args.photos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
