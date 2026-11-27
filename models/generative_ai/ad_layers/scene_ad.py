"""상품 장면을 먼저 만들고 실제 화면에 맞춰 글자 레이어를 배치합니다."""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps

from .creative_brief import TEXT_ROLES, ask_stage, plan_copy, validate_boxes
from .layers import compose, load_settings, save_layout, to_float
from .text_render import render_texts


LAYOUT = """Design the typesetting on the supplied finished photograph. Work with the actual visible product
and negative space. Use the measured product bounds to locate the subject. Keep each text block off the product
and away from busy details and frame edges. Use the focus from the brief to lead the eye, with the other roles
clearly subordinate. Group related lines and use consistent alignment and sufficient contrast. Keep the wording.
Return JSON with "boxes" and "styles", both keyed by headline, product_name, and price.
Boxes are integer [left, top, width, height] percentages of the image. Make them big enough to render the full copy.
For each style return "font" (one of the supplied fonts), "align" (left, center, right), "color" (#RRGGBB),
"size" (intended font size as an integer percentage of canvas width), and "max_lines" (1 or 2 for the
headline or product_name, 1 for price). If a prior layout failed, fix the reported issue on this same photograph."""


def layout_texts(answer, copy, foreground, fonts):
    boxes = validate_boxes(answer.get("boxes"), TEXT_ROLES)
    styles = answer.get("styles")
    if not isinstance(styles, dict) or styles.keys() != boxes.keys():
        raise ValueError("문구마다 글꼴, 정렬, 색과 크기가 필요합니다")
    height, width = foreground.shape
    texts = []
    for role, text in copy.items():
        left, top, box_width, box_height = boxes[role]
        x, y = round(left * width / 100), round(top * height / 100)
        w = round((left + box_width) * width / 100) - x
        h = round((top + box_height) * height / 100) - y
        if foreground[y:y + h, x:x + w].any():
            raise ValueError(f"{role} 영역이 실제 제품과 겹쳤습니다")
        style = styles[role]
        if not isinstance(style, dict):
            raise ValueError(f"{role} 글자 스타일이 빠졌습니다")
        font, align, color = (style.get(key) for key in ("font", "align", "color"))
        size, max_lines = style.get("size"), style.get("max_lines")
        if (font not in tuple(fonts) or align not in ("left", "center", "right")
                or not isinstance(color, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", color)
                or type(size) is not int or not 2 <= size <= 20
                or type(max_lines) is not int or not 1 <= max_lines <= (1 if role == "price" else 2)):
            raise ValueError(f"{role} 글자 스타일 값이 올바르지 않습니다")
        texts.append({"role": role, "text": text, "font": font, "align": align, "color": color,
                      "x": x, "y": y, "w": w, "h": h, "file": f"text_{role}.png", "max_lines": max_lines,
                      "max_font_size": round(size * width / 100), "min_font_size": max(6, round(width * 0.02))})
    return texts


def make_scene_ad(photo, fields, idea, models, folder):
    from ..photo_retouch.product_mask import product_mask

    if folder.exists():
        raise FileExistsError(folder)
    settings = load_settings()
    ai, model = models["ai"], models["generation"]
    brief = plan_copy(ai, photo, fields, idea, scene_first=True)
    copy = {"headline": brief["headline"], "product_name": fields["product_name"],
            "price": f"{fields['price_krw']:,}원"}
    canvas = settings["canvas"]
    prompt = ("Use the first image as the product reference and create the photograph on the second canvas. "
              f"{brief['scene']} Leave a generous, quiet area at the {brief['copy_space']} for later typography. "
              "Keep the product and props clear of that area. Preserve the actual package and its labels. "
              "Do not add any advertising text, new labels or graphic overlays to the photograph.")
    model.use_lora(None, 0)
    generation = settings["generation"]
    blank = Image.new("RGB", (canvas["width"], canvas["height"]), canvas["background_color"])
    scene = model.edit_with_reference(photo, blank, prompt, generation["seed"], generation["steps"]).convert("RGB")
    pixels = to_float(scene)
    foreground = product_mask(models["mask"], pixels) > settings["product"]["mask_threshold"]
    rows, columns = np.nonzero(foreground)
    if not len(rows):
        raise ValueError("생성한 장면에서 글자로 가리면 안 되는 제품 영역을 찾지 못했습니다")
    bounds = [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1]
    layout_input = {"focus": brief["focus"], "message": brief["message"], "copy": copy,
                    "image_size_pixels": list(scene.size), "product_bounds_pixels": bounds,
                    "fonts": list(settings["fonts"])}

    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as staging:
        result = Path(staging)
        answer = ask_stage(ai, "ad_layout", LAYOUT, layout_input, [pixels])

        def render(answer):
            texts = layout_texts(answer, copy, foreground, settings["fonts"])
            render_texts(texts, scene.size, result)
            return texts

        try:
            texts = render(answer)
        except ValueError as error:
            retry = layout_input | {"previous_boxes": answer.get("boxes"), "layout_error": str(error)}
            texts = render(ask_stage(ai, "ad_layout", LAYOUT, retry, [pixels]))

        layout = {"size": list(scene.size), "background": {"file": "scene.png"}, "fields": fields,
                  "brief": brief, "scene_prompt": prompt, "product_bounds_pixels": bounds, "texts": texts}
        scene.save(result / "scene.png")
        compose(layout, result).save(result / "ad.png")
        save_layout(layout, result)
        result.rename(folder)
    return layout


def main():
    from .make_ad import load_models

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items", type=Path, help="파일명, 장면 한 줄, 상품 정보를 담은 JSON")
    parser.add_argument("photos", type=Path, help="참고 사진 폴더")
    parser.add_argument("output", type=Path, help="상품별 장면과 글자 레이어를 저장할 폴더")
    args = parser.parse_args()
    items = json.loads(args.items.read_text(encoding="utf-8"))
    models = load_models()
    for index, item in enumerate(items, 1):
        reference = Path(item["file"])
        folder = args.output / f"{index:02d}_{reference.stem}"
        with Image.open(args.photos / reference) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        make_scene_ad(photo, item["fields"], item.get("idea", ""), models, folder)
        print(folder / "ad.png", flush=True)


if __name__ == "__main__":
    main()
