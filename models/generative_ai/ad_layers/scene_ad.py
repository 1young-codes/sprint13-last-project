"""포스터의 구도를 먼저 정하고 그림과 글자를 별도 레이어로 만듭니다."""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .creative_brief import LAYOUT_ROLES, TEXT_ROLES, ask_stage, plan_copy, required_text, validate_boxes
from .layers import compose, load_layout, load_settings, save_layout, to_float
from .text_render import render_texts


DESIGN = """Art-direct one complete Korean campaign poster before any artwork is generated. The first image
shows the actual product. Remaining images, if present, are references for hierarchy, rhythm, and the relationship
between photo, type, and color. Borrow design principles only. Never copy their text, brands, or artwork.
Use the real product details and campaign message to make one visible creative choice that ties together the
photography, graphic color and typography. Decide what wins the first glance. Give supporting copy distinct
scale and grouping. Use deliberately shaped negative space and allow type beside or over quiet photo areas when
it supports the concept. Avoid the catalog pattern of a centered rectangular image with a heading above and a
name/price footer. Use one visual decision that still reads at thumbnail size, such as a close-up of a real selling
detail or a strong axis linking photo and copy. A continuous full-bleed photograph can provide the whole poster
surface. Place the two supporting text elements close enough to read as a group. Use a solid color accent when
it connects the composition or groups text. Give the main line room to lead in scale and contrast."""

DESIGN_SCHEMA = """A design object has "concept" (the visual idea and why it serves the message), "focus" (product, headline,
product_name, or price), "background" (#RRGGBB), "accent" (null, or an object with "shape" rectangle or ellipse,
"box", and "color" #RRGGBB), "artwork_box", "edge" (blend or framed), "crop" (preserve or intentional), and "art_direction" (English
photographic framing, lighting, materials and background for the artwork region; no text or graphic overlays).
Use the blended edge for a calm photo perimeter that can merge into a plain graphic field. Keep any solid graphic
accent away from that seam. A framed edge can align a deliberate crop with the type or cover the whole canvas.
If there is a quiet area for type over the artwork, say where it is and describe its light and texture in art_direction.
Include "boxes" and "styles", both keyed by headline, product_name and price.
All boxes are [left, top, width, height] as whole percentages of the poster from its top-left. Keep text boxes
inside the canvas, off one another, and large enough for the supplied copy. Text may overlap the artwork box
only where you plan quiet background. Group related text by alignment and use sufficient contrast.
For each style return "font" (one of the supplied fonts), "align" (left, center, right), "color" (#RRGGBB),
"size" (intended font size as an integer percentage of canvas width), and "max_lines" (1 or 2 for the
headline or product_name, 1 for price). If a prior design failed, address the reported issue."""

REVIEW = """Review the rendered poster draft, shown first, against the composition principles of the following
reference posters. The draft uses a temporary source photo. Judge the hierarchy, balance, scale, and connection
between the photographic subject, type, and color using their visible positions. Ignore photography changes that
will occur during final generation. Pay particular attention to whether this is still a catalog card with a central
rectangle and unrelated labels or decoration. If the poster needs improvement, revise this same poster using
the complete original design schema; keep its product, message and exact copy. Return JSON with "reason"
(one sentence explaining the observation) and "revision" (null to keep, or the complete revised design)."""

FINAL_REVIEW = """Review the first poster, using the second image (its generated photo without type) and the reference
posters that follow. Keep the generated photo, position, crop and canvas color. Use the actual product bounds to
place type outside the product. Group the related text, give the main line a distinct scale, and judge contrast
using the actual photograph. Match the lettering to the visual mood and give multi-line Korean copy a natural
line break. Use the supplied wording exactly. Return JSON with reason and revision (null or an
object containing boxes, styles and accent per the design schema). Fix any reported validation error."""

COMPARE = """Pick the stronger Korean ad. The first two images show the same photo and exact copy in different
layouts. Remaining images are composition references. Prefer the second only if its hierarchy, grouping or
legibility improves noticeably while preserving product visibility. Return JSON with winner (1 or 2) and reason."""


def is_color(value):
    return isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value) is not None


def pixel_box(box, size):
    left, top, width, height = box
    canvas_w, canvas_h = size
    x, y = round(left * canvas_w / 100), round(top * canvas_h / 100)
    return (x, y, round((left + width) * canvas_w / 100) - x,
            round((top + height) * canvas_h / 100) - y)


def layout_texts(answer, copy, canvas_size, fonts):
    boxes = validate_boxes(answer.get("boxes"), TEXT_ROLES)
    styles = answer.get("styles")
    if not isinstance(styles, dict) or styles.keys() != boxes.keys():
        raise ValueError("문구마다 글꼴, 정렬, 색과 크기가 필요합니다")
    width = canvas_size[0]
    texts = []
    for role, text in copy.items():
        x, y, w, h = pixel_box(boxes[role], canvas_size)
        style = styles[role]
        if not isinstance(style, dict):
            raise ValueError(f"{role} 글자 스타일이 빠졌습니다")
        font, align, color = (style.get(key) for key in ("font", "align", "color"))
        size, max_lines = style.get("size"), style.get("max_lines")
        if (font not in tuple(fonts) or align not in ("left", "center", "right")
                or not is_color(color)
                or type(size) is not int or not 2 <= size <= 20
                or type(max_lines) is not int or not 1 <= max_lines <= (1 if role == "price" else 2)):
            raise ValueError(f"{role} 글자 스타일 값이 올바르지 않습니다")
        texts.append({"role": role, "text": text, "font": font, "align": align, "color": color,
                      "x": x, "y": y, "w": w, "h": h, "file": f"text_{role}.png", "max_lines": max_lines,
                      "max_font_size": round(size * width / 100), "min_font_size": max(6, round(width * 0.02))})
    return texts


def design_layers(answer, copy, size, fonts, folder):
    required_text(answer, "concept", "포스터 구도")
    required_text(answer, "art_direction", "포스터 구도")
    accent = answer.get("accent")
    if (answer.get("focus") not in LAYOUT_ROLES or answer.get("crop") not in ("preserve", "intentional")
            or answer.get("edge") not in ("blend", "framed")
            or not is_color(answer.get("background"))):
        raise ValueError("포스터 구도에 강조점, 제품 잘림 정책과 배경색이 필요합니다")
    validate_boxes({"artwork": answer.get("artwork_box")}, ("artwork",))
    x, y, w, h = pixel_box(answer["artwork_box"], size)
    if min(answer["artwork_box"][2:]) < 35:
        raise ValueError("상품 사진을 보여 줄 영역이 너무 작습니다")
    background = Image.new("RGB", size, answer["background"])
    if accent is not None:
        if (not isinstance(accent, dict) or accent.get("shape") not in ("rectangle", "ellipse")
                or not is_color(accent.get("color"))):
            raise ValueError("색면의 형태와 색이 올바르지 않습니다")
        validate_boxes({"accent": accent.get("box")}, ("accent",))
        ax, ay, aw, ah = pixel_box(accent["box"], size)
        if (accent["color"].lower() == answer["background"].lower()
                or (x <= ax and y <= ay and x + w >= ax + aw and y + h >= ay + ah)):
            raise ValueError("색면이 배경과 구분되고 상품 사진 밖에서 보이는 영역이 필요합니다")
        draw = ImageDraw.Draw(background)
        shape = draw.rectangle if accent["shape"] == "rectangle" else draw.ellipse
        shape((ax, ay, ax + aw - 1, ay + ah - 1), fill=accent["color"])
    texts = layout_texts(answer, copy, size, fonts)
    render_texts(texts, size, folder)
    background.save(folder / "background.png")
    return {"size": list(size), "background": {"file": "background.png"}, "texts": texts,
            "artwork": {"file": "scene.png", "x": x, "y": y, "w": w, "h": h}}


def find_foreground(scene, mask_model, threshold, crop):
    from ..photo_retouch.product_mask import product_mask

    foreground = product_mask(mask_model, to_float(scene)) > threshold
    rows, columns = np.nonzero(foreground)
    if not len(rows):
        raise ValueError("생성한 사진에서 제품을 찾지 못했습니다")
    left, top, right, bottom = columns.min(), rows.min(), columns.max() + 1, rows.max() + 1
    if crop == "preserve" and (left == 0 or top == 0 or right == scene.width or bottom == scene.height):
        raise ValueError("생성한 사진에서 제품이 가장자리에 잘렸습니다")

    return foreground


def check_text_overlap(foreground, layout, folder):
    mask = np.zeros(tuple(reversed(layout["size"])), dtype=bool)
    x, y = layout["artwork"]["x"], layout["artwork"]["y"]
    height, width = foreground.shape
    mask[y:y + height, x:x + width] = foreground
    for text in layout["texts"]:
        with Image.open(folder / text["file"]) as layer:
            ink = np.asarray(layer.getchannel("A")) > 128
        x, y = text["x"], text["y"]
        if (mask[y:y + text["h"], x:x + text["w"]] & ink).any():
            raise ValueError(f"{text['role']} 글자가 실제 제품을 가립니다")


def review_finished(ai, answer, layout, scene, foreground, folder, design_input, references):
    """실제 사진을 그대로 써서 문구 배치를 확인하고, 검사를 통과한 수정안만 반영합니다."""
    if not references:
        check_text_overlap(foreground, layout, folder)
        return answer, layout, None
    try:
        check_text_overlap(foreground, layout, folder)
    except ValueError as error:
        original_error = str(error)
    else:
        original_error = None
    first = to_float(compose(layout, folder))
    artwork = layout["artwork"]
    size = tuple(layout["size"])
    rows, columns = np.nonzero(foreground)
    placement = [round(100 * value / limit) for value, limit in zip(
        (artwork["x"] + columns.min(), artwork["y"] + rows.min(), artwork["x"] + columns.max() + 1,
         artwork["y"] + rows.max() + 1), (*size, *size), strict=True)]
    inputs = design_input | {"design": answer, "product_bounds_left_top_right_bottom": placement,
                             "layout_error": original_error}
    for _ in range(2):
        review = ask_stage(ai, "ad_final_layout", FINAL_REVIEW + "\n" + DESIGN_SCHEMA, inputs,
                           [first, to_float(scene), *references])
        revision = review.get("revision")
        if not isinstance(revision, dict):
            break
        revised = answer | {key: revision[key] for key in ("boxes", "styles", "accent") if key in revision}
        if revised == answer:
            break
        with TemporaryDirectory(dir=folder) as staging:
            candidate = Path(staging)
            try:
                updated = design_layers(revised, design_input["copy"], size, design_input["fonts"], candidate)
                check_text_overlap(foreground, updated, candidate)
            except ValueError as error:
                inputs = inputs | {"design": revised, "layout_error": str(error)}
                continue
            scene.save(candidate / artwork["file"])
            selection = ask_stage(ai, "ad_final_compare", COMPARE,
                                  {"copy": design_input["copy"], "first_layout_error": original_error},
                                  [first, to_float(compose(updated, candidate)), *references])
            if selection.get("winner") != 2:
                break
            for path in candidate.iterdir():
                path.replace(folder / path.name)
            return revised, updated, selection.get("reason")
    if original_error:
        raise ValueError(original_error)
    return answer, layout, None


def blend_edges(scene, foreground=None, *, position=(0, 0), canvas_size=None):
    feather = max(1, round(min(scene.size) * 0.16))
    x, y = np.arange(scene.width), np.arange(scene.height)
    left, top = position
    right, bottom = (left + scene.width, top + scene.height)
    # 캔버스 끝에는 이웃하는 배경이 없으므로 사진을 그대로 둡니다.
    horizontal = np.minimum(x if canvas_size is None or left > 0 else np.full_like(x, feather),
                            scene.width - 1 - x if canvas_size is None or right < canvas_size[0]
                            else np.full_like(x, feather))
    vertical = np.minimum(y if canvas_size is None or top > 0 else np.full_like(y, feather),
                          scene.height - 1 - y if canvas_size is None or bottom < canvas_size[1]
                          else np.full_like(y, feather))
    distance = np.minimum(horizontal[None, :], vertical[:, None])
    matte = (np.clip(distance / feather, 0, 1) * 255).astype(np.uint8)
    if foreground is not None:
        matte[foreground] = 255
    scene = scene.convert("RGBA")
    scene.putalpha(Image.fromarray(matte))
    return scene


def preview_artwork(photo, artwork, canvas_size, design):
    picture_size = (artwork["w"], artwork["h"])
    x, y = artwork["x"], artwork["y"]
    if design["crop"] == "intentional":
        picture = ImageOps.fit(photo, picture_size, Image.Resampling.LANCZOS)
        return (blend_edges(picture, position=(x, y), canvas_size=canvas_size)
                if design["edge"] == "blend" else picture)
    if design["edge"] == "framed":
        return ImageOps.pad(photo, picture_size, Image.Resampling.LANCZOS, color=design["background"])
    picture = ImageOps.contain(photo, picture_size, Image.Resampling.LANCZOS)
    offset_x, offset_y = (artwork["w"] - picture.width) // 2, (artwork["h"] - picture.height) // 2
    picture = blend_edges(picture, position=(x + offset_x, y + offset_y), canvas_size=canvas_size)
    area = Image.new("RGBA", picture_size)
    area.alpha_composite(picture, (offset_x, offset_y))
    return area


def make_scene_ad(photo, fields, idea, models, folder, *, references=(), preview=False, proposal=None):

    if folder.exists():
        raise FileExistsError(folder)
    settings = load_settings()
    ai = models["ai"]
    if proposal and proposal["fields"] != fields:
        raise ValueError("시안의 상품 정보가 현재 입력과 다릅니다")
    brief = proposal["brief"] if proposal else plan_copy(ai, photo, fields, idea)
    copy = {"headline": brief["headline"], "product_name": fields["product_name"],
            "price": f"{fields['price_krw']:,}원"}
    canvas = settings["canvas"]
    size = (canvas["width"], canvas["height"])
    design_input = {"brief": brief, "copy": copy, "fonts": list(settings["fonts"]), "canvas_size": size}
    images = [to_float(photo), *map(to_float, references)]

    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as staging:
        result = Path(staging)
        design_task = DESIGN + "\nReturn a JSON design object.\n" + DESIGN_SCHEMA
        answer = proposal["design"] if proposal else ask_stage(ai, "ad_layout", design_task, design_input, images)
        try:
            layout = design_layers(answer, copy, size, settings["fonts"], result)
        except ValueError as error:
            if proposal:
                raise
            retry = design_input | {"previous_design": answer, "layout_error": str(error)}
            answer = ask_stage(ai, "ad_layout", design_task, retry, images)
            layout = design_layers(answer, copy, size, settings["fonts"], result)

        review_reason = proposal.get("review_reason") if proposal else None
        if references and not proposal:
            artwork = layout["artwork"]
            preview_artwork(photo, artwork, size, answer).save(result / artwork["file"])
            review = ask_stage(ai, "ad_layout_review", REVIEW + "\n" + DESIGN_SCHEMA,
                               design_input | {"design": answer}, [to_float(compose(layout, result)), *images[1:]])
            revised = review.get("revision")
            review_reason = review.get("reason")
            if isinstance(revised, dict):
                try:
                    layout = design_layers(revised, copy, size, settings["fonts"], result)
                except ValueError as error:
                    review_reason = f"배치 수정 보류 · {error}"
                else:
                    answer = revised

        artwork = layout["artwork"]
        picture_size = (artwork["w"], artwork["h"])
        prompt = ("Use the first image as the product reference and create the commissioned photograph on the second canvas. "
                  f"{answer['art_direction']} Preserve the recognizable product details and any visible packaging and labels. "
                  "Add no advertising text or overlays.")
        if answer["edge"] == "blend":
            prompt += f" Keep the outer background low-detail and close to the surrounding {answer['background']} color."
        if answer["crop"] == "preserve":
            prompt += " Keep all of the product in frame with space around its edges."
        if preview:
            scene = preview_artwork(photo, artwork, size, answer)
            final_review_reason = None
        else:
            model = models["generation"]
            model.use_lora(None, 0)
            generation = settings["generation"]
            blank = Image.new("RGB", picture_size, answer["background"])
            generated = model.edit_with_reference(photo, blank, prompt, generation["seed"], generation["steps"])
            scene = ImageOps.fit(generated.convert("RGB"), picture_size, Image.Resampling.LANCZOS)
            foreground = find_foreground(scene, models["mask"], settings["product"]["mask_threshold"], answer["crop"])
            if answer["edge"] == "blend":
                scene = blend_edges(scene, foreground, position=(artwork["x"], artwork["y"]), canvas_size=size)
            scene.save(result / artwork["file"])
            answer, layout, final_review_reason = review_finished(ai, answer, layout, scene, foreground, result,
                                                                  design_input, images[1:])
        layout.update(fields=fields, brief=brief, design=answer, scene_prompt=prompt, preview=preview,
                      review_reason=review_reason, final_review_reason=final_review_reason)
        if preview:
            scene.save(result / "scene.png")
        compose(layout, result).save(result / "ad.png")
        save_layout(layout, result)
        result.rename(folder)
    return layout


def main():
    from .make_ad import load_models
    from ..photo_retouch.interpret import Interpreter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items", type=Path, help="파일명, 장면 한 줄, 상품 정보를 담은 JSON")
    parser.add_argument("photos", type=Path, help="참고 사진 폴더")
    parser.add_argument("output", type=Path, help="상품별 장면과 글자 레이어를 저장할 폴더")
    parser.add_argument("--references", nargs="+", type=Path, default=[], metavar="PHOTO", help="포스터 구도 참고 이미지")
    parser.add_argument("--preview", action="store_true", help="원본 사진으로 구도 시안 만들기")
    parser.add_argument("--designs", type=Path, help="상품별로 저장된 구도 시안을 그대로 사용하기")
    parser.add_argument("--low-vram", action="store_true", help="가중치를 나눠 올려 그래픽 메모리를 절약하기")
    args = parser.parse_args()
    items = json.loads(args.items.read_text(encoding="utf-8"))
    references = []
    for path in args.references:
        with Image.open(path) as source:
            references.append(ImageOps.exif_transpose(source).convert("RGB"))
    models = {"ai": Interpreter()} if args.preview else load_models(low_vram=args.low_vram)
    for index, item in enumerate(items, 1):
        reference = Path(item["file"])
        folder = args.output / f"{index:02d}_{reference.stem}"
        with Image.open(args.photos / reference) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        proposal = load_layout(args.designs / folder.name) if args.designs else None
        make_scene_ad(photo, item["fields"], item.get("idea", ""), models, folder,
                      references=references, preview=args.preview, proposal=proposal)
        print(folder / "ad.png", flush=True)


if __name__ == "__main__":
    main()
