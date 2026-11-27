"""포스터의 구도를 먼저 정하고 그림과 글자를 별도 레이어로 만듭니다."""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .creative_brief import LAYOUT_ROLES, TEXT_ROLES, ask_stage, plan_copy, required_text, validate_boxes
from .layers import compose, cutout_product, load_layout, load_settings, save_layout, to_float
from .text_render import render_texts


DESIGN = """Art-direct one complete Korean campaign poster before any artwork is generated. The first image
shows the actual product. Remaining images, if present, are references for hierarchy, rhythm, and the relationship
between photo, type, and color. Borrow design principles only. Never copy their text, brands, or artwork.
Use the real product details and campaign message to make one visible creative choice that ties together the
photography, graphic color and typography. Decide what wins the first glance. Give supporting copy distinct
scale and grouping. Use deliberately shaped negative space and allow type beside or over quiet photo areas when
it supports the concept. Avoid the catalog pattern of a centered rectangular image with a heading above and a
name/price footer. Use one visual decision that still reads at thumbnail size, such as a close-up of a real selling
detail or a strong axis linking photo and copy. Start with a shared photographic field where light or shape can
lead from the product into the copy. Use a separate graphic panel only when its edge or color plays an intentional
part in that direction. Place the two supporting text elements close enough to read as a group. Use a solid color accent only
if it continues a line or a color from the photograph, or gives supporting type an intentional shared field.
An accent can cross from the edge or sit in quiet space over the photo. Give the main line room to lead."""

DESIGN_SCHEMA = """A design object has "concept" (the visual idea and why it serves the message), "focus" (product, headline,
product_name, or price), "background" (#RRGGBB), "accent" (null, or an object with "shape" rectangle or ellipse,
"box", "color" #RRGGBB, and "layer" background or overlay), "artwork_box", "subject_box", "edge" (blend or framed),
"crop" (preserve or intentional), and "art_direction" (English
photographic framing, lighting, materials and background for the artwork region; no text or graphic overlays).
Use a background accent outside the artwork, or an overlay accent over quiet space without covering the product.
Use the blended edge for a calm photo perimeter that can merge into a plain graphic field. A framed edge can
align a deliberate crop with the type or cover the whole canvas.
Use preserve for a complete product by default. Choose intentional only when a motivated crop emphasizes a real
detail and the shopper can still identify the product and any important label.
If there is a quiet area for type over the artwork, say where it is and describe its light and texture in art_direction.
Include "boxes" and "styles", both keyed by headline, product_name and price.
All boxes are [left, top, width, height]. Use whole percentages of the poster from its top-left, except subject_box,
which is relative to the photograph itself. Put the product inside subject_box with the intended scale and space
around the edges. In the shared photo area, reserve space for the headline and the other copy outside the product.
Keep text boxes
inside the canvas, off one another, and large enough for the supplied copy. Text may overlap the artwork box
only where you plan quiet background. Group related text by alignment and use sufficient contrast.
For each style return "font" (one of the supplied fonts), "align" (left, center, right), "color" (#RRGGBB),
"size" (intended font size as an integer percentage of canvas width), and "max_lines" (1 or 2 for the
headline or product_name, 1 for price). If two lines are intentional, set "break_after" to the number of
space-separated words on the first line. Keep Korean phrases, quantities and product names intact at the
break. If a prior design failed, address the reported issue."""

FINAL_REVIEW = """Review the first poster, using the second image (its generated photo without type) and the reference
posters that follow. Keep the generated photo, position, crop and canvas color. Use the actual product bounds to
place type outside the product. Group the related text, give the main line a distinct scale, and judge contrast
using the actual photograph. Match the lettering to the visual mood. Set break_after for any specific Korean
line break you intend. Use the supplied wording exactly. Return JSON with reason and revision (null or an
object containing boxes, styles and accent per the design schema). Fix any reported validation error."""

CRITIQUE = """The first image is the previous finished ad, the second is the source product, and the rest are
design references. Identify the most consequential visible weakness of the ad that changing the photograph
could fix. Ground it in details you see. Propose one new visual device justified by a real detail of the product
or the campaign message. Learn from the references how the main subject and a decisive choice of scale, light,
material or type make an idea visible. A brighter background or a tidier product shot alone is insufficient.
Preserve product identity and visual presence. Borrow principles only. Return JSON
with 'weakness' (in Korean) and 'hypothesis' (an English art direction idea)."""

PHOTO_DIRECTION = """Write the English instruction that the image model will receive for this photograph.
The image is the source product. Make the design concept and any revision hypothesis visibly happen in the photograph.
Use the approved artwork box and copy positions to describe the subject and quiet space relative to the artwork.
Describe the camera framing, the exact visible effect, and how its light, material or gesture leads towards the
planned type. Keep the product prominent and recognizable, with its visible details, packaging and labels.
When there is a product demonstration, describe exactly what supports each object and the physical relationship
of the moving parts. Use surroundings and props only where they serve this particular image or the owner's
request. Follow the agreed crop. State only what the generated photograph should show. Include no ad text,
graphics, coordinates or layout markup. Return JSON with one string, 'scene'."""


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
        size, max_lines, break_after = (style.get(key) for key in ("size", "max_lines", "break_after"))
        if (font not in tuple(fonts) or align not in ("left", "center", "right")
                or not is_color(color)
                or type(size) is not int or not 2 <= size <= 20
                or type(max_lines) is not int or not 1 <= max_lines <= (1 if role == "price" else 2)):
            raise ValueError(f"{role} 글자 스타일 값이 올바르지 않습니다")
        layer = {"role": role, "text": text, "font": font, "align": align, "color": color,
                 "x": x, "y": y, "w": w, "h": h, "file": f"text_{role}.png", "max_lines": max_lines,
                 "max_font_size": round(size * width / 100), "min_font_size": max(6, round(width * 0.02))}
        if break_after is not None:
            if max_lines != 2 or type(break_after) is not int or not 0 < break_after < len(text.split()):
                raise ValueError(f"{role} 줄바꿈 위치가 올바르지 않습니다")
            layer["break_after"] = break_after
        texts.append(layer)
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
    if "subject_box" in answer:
        validate_boxes({"subject": answer["subject_box"]}, ("subject",))
    x, y, w, h = pixel_box(answer["artwork_box"], size)
    if min(answer["artwork_box"][2:]) < 35:
        raise ValueError("상품 사진을 보여 줄 영역이 너무 작습니다")
    background = Image.new("RGB", size, answer["background"])
    graphic = None
    if accent is not None:
        if (not isinstance(accent, dict) or accent.get("shape") not in ("rectangle", "ellipse")
                or accent.get("layer", "background") not in ("background", "overlay")
                or not is_color(accent.get("color"))):
            raise ValueError("색면의 형태와 색이 올바르지 않습니다")
        validate_boxes({"accent": accent.get("box")}, ("accent",))
        ax, ay, aw, ah = pixel_box(accent["box"], size)
        overlay = accent.get("layer") == "overlay"
        if not overlay and (accent["color"].lower() == answer["background"].lower()
                            or (x <= ax and y <= ay and x + w >= ax + aw and y + h >= ay + ah)):
            raise ValueError("배경 색면은 바탕과 구분되고 사진 밖에서 보여야 합니다")
        target = Image.new("RGBA", (aw, ah)) if overlay else background
        left, top = (0, 0) if overlay else (ax, ay)
        draw = ImageDraw.Draw(target)
        shape = draw.rectangle if accent["shape"] == "rectangle" else draw.ellipse
        shape((left, top, left + aw - 1, top + ah - 1), fill=accent["color"])
        if overlay:
            graphic = {"file": "graphic.png", "x": ax, "y": ay, "w": aw, "h": ah}
            target.save(folder / graphic["file"])
    texts = layout_texts(answer, copy, size, fonts)
    render_texts(texts, size, folder)
    background.save(folder / "background.png")
    layout = {"size": list(size), "background": {"file": "background.png"}, "texts": texts,
              "artwork": {"file": "scene.png", "x": x, "y": y, "w": w, "h": h}}
    if graphic:
        layout["graphic"] = graphic
    return layout


def product_reference(photo, mask_model, threshold, picture_size, subject_box):
    from ..photo_retouch.product_mask import product_mask

    mask = product_mask(mask_model, to_float(photo))
    if not (mask > threshold).any():
        return photo
    product = cutout_product(photo, mask, threshold)
    if subject_box:
        x, y, width, height = pixel_box(subject_box, picture_size)
        product = ImageOps.contain(product, (width, height), Image.Resampling.LANCZOS)
        reference = Image.new("RGB", picture_size, "#808080")
        reference.paste(product, (x + (width - product.width) // 2, y + (height - product.height) // 2), product)
        return reference
    margin = round(max(product.size) / 10)
    reference = Image.new("RGB", (product.width + 2 * margin, product.height + 2 * margin), "#808080")
    reference.paste(product, (margin, margin), product)
    return reference


def find_foreground(scene, mask_model, threshold):
    from ..photo_retouch.product_mask import product_mask

    foreground = product_mask(mask_model, to_float(scene)) > threshold
    if not foreground.any():
        raise ValueError("생성한 사진에서 제품을 찾지 못했습니다")
    return foreground


def check_text_overlap(foreground, layout, folder):
    mask = np.zeros(tuple(reversed(layout["size"])), dtype=bool)
    x, y = layout["artwork"]["x"], layout["artwork"]["y"]
    height, width = foreground.shape
    mask[y:y + height, x:x + width] = foreground
    layers = [(f"{text['role']} 글자", text) for text in layout["texts"]]
    if graphic := layout.get("graphic"):
        layers.append(("색면", graphic))
    for name, layer_info in layers:
        with Image.open(folder / layer_info["file"]) as layer:
            ink = np.asarray(layer.getchannel("A")) > 128
        x, y = layer_info["x"], layer_info["y"]
        if (mask[y:y + layer_info["h"], x:x + layer_info["w"]] & ink).any():
            raise ValueError(f"{name} 레이어가 실제 제품을 가립니다")


def review_finished(ai, answer, layout, scene, foreground, folder, design_input, references):
    """완성 사진에서 레이어가 피사체를 가리는 경우에만 배치를 고칩니다."""
    try:
        check_text_overlap(foreground, layout, folder)
    except ValueError as error:
        original_error = str(error)
    else:
        return answer, layout, None
    first = compose(layout, folder)
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
                           [to_float(first), to_float(scene), *references])
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
                scene.save(candidate / artwork["file"])
                check_text_overlap(foreground, updated, candidate)
            except ValueError as error:
                inputs = inputs | {"design": revised, "layout_error": str(error)}
                continue
            first.save(folder / "before-typesetting.png")
            for path in candidate.iterdir():
                path.replace(folder / path.name)
            if "graphic" in layout and "graphic" not in updated:
                (folder / layout["graphic"]["file"]).unlink()
            return revised, updated, review.get("reason")
    raise ValueError(original_error)


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


def make_scene_ad(photo, fields, idea, models, folder, *, references=(), preview=False, proposal=None, previous=None,
                  rejected=None, feedback=None):

    if folder.exists():
        raise FileExistsError(folder)
    settings = load_settings()
    ai = models["ai"]
    if proposal and previous:
        raise ValueError("저장한 시안 적용과 완성본 개선 중 하나를 선택하세요")
    if rejected and not previous:
        raise ValueError("보류한 후보를 전달하려면 비교 기준이 되는 완성본도 필요합니다")
    prior = load_layout(previous) if previous else None
    kept = prior or proposal
    if kept and kept["fields"] != fields:
        raise ValueError("시안의 상품 정보가 현재 입력과 다릅니다")
    brief = kept["brief"] if kept else plan_copy(ai, photo, fields, idea)
    copy = {"headline": brief["headline"], "product_name": fields["product_name"],
            "price": f"{fields['price_krw']:,}원"}
    canvas = settings["canvas"]
    size = (canvas["width"], canvas["height"])
    images = [to_float(photo), *map(to_float, references)]
    revision_info = None
    if prior:
        if prior.get("preview"):
            raise ValueError("사진 생성이 끝난 광고만 개선에 사용할 수 있습니다")
        with Image.open(previous / "ad.png") as finished:
            prior_image = to_float(finished)
        previous_design = prior.get("design", {})
        inputs = {"brief": brief, "copy": copy, "previous_concept": previous_design.get("concept", brief["message"])}
        inputs["previous_art_direction"] = previous_design.get("art_direction", prior.get("scene_prompt"))
        inputs["previous_visual_review"] = feedback or prior.get("revision", {}).get("visual_review")
        critique_prompt, critique_images = CRITIQUE, [prior_image, *images]
        if rejected:
            discarded = load_layout(rejected)
            if discarded["fields"] != fields:
                raise ValueError("보류한 후보의 상품 정보가 현재 입력과 다릅니다")
            if discarded.get("preview"):
                raise ValueError("사진 생성이 끝난 광고만 보류한 후보로 사용할 수 있습니다")
            inputs["rejected_attempt"] = {"art_direction": discarded["scene_prompt"],
                                          "visual_review": discarded.get("revision", {}).get("visual_review")}
            with Image.open(rejected / "ad.png") as rejected_image:
                critique_images.append(to_float(rejected_image))
            critique_prompt += "\nThe last image is a rejected attempt. Address its weakness and avoid repeating it."
        criticism = ask_stage(ai, "ad_cycle_critique", critique_prompt, inputs, critique_images)
        revision_info = {key: required_text(criticism, key, "완성본 검토") for key in ("weakness", "hypothesis")}
    design_input = {"brief": brief, "copy": copy, "fonts": list(settings["fonts"]), "canvas_size": size}
    if revision_info:
        design_input.update(brief={"message": brief["message"], "evidence": brief.get("evidence")},
                            idea=idea, revision_feedback=revision_info)

    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as staging:
        result = Path(staging)
        design_task = DESIGN + "\nReturn a JSON design object.\n" + DESIGN_SCHEMA
        if revision_info:
            design_task += ("\nMake room for the visual idea in revision_feedback. Give the photography "
                            "and type a new shared composition with the same exact copy.")
        answer = proposal["design"] if proposal else ask_stage(ai, "ad_layout", design_task, design_input, images)
        try:
            layout = design_layers(answer, copy, size, settings["fonts"], result)
        except ValueError as error:
            if proposal:
                raise
            retry = design_input | {"previous_design": answer, "layout_error": str(error)}
            answer = ask_stage(ai, "ad_layout", design_task, retry, images)
            layout = design_layers(answer, copy, size, settings["fonts"], result)

        if not proposal:
            direction_input = {"message": brief["message"], "idea": idea, "design": answer}
            if revision_info:
                direction_input.update(revision_info)
            direction = ask_stage(ai, "ad_scene_direction", PHOTO_DIRECTION, direction_input, images[:1])
            answer = answer | {"art_direction": required_text(direction, "scene", "촬영 연출")}

        artwork = layout["artwork"]
        picture_size = (artwork["w"], artwork["h"])
        subject_box = answer.get("subject_box")
        placement = ("Follow the reference's product placement and scale if it has a plain gray background. "
                     if subject_box else "")
        prompt = ("Make a new commissioned photograph with a newly built setting and lighting. "
                  f"{placement}Replace the surroundings with the requested setting. Keep the product shape, material and details. "
                  f"{answer['art_direction']} Preserve the recognizable product details and genuine packaging labels. "
                  "Keep surroundings, props and surfaces free of stray lettering, numbers and logos. "
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
            threshold = settings["product"]["mask_threshold"]
            reference = product_reference(photo, models["mask"], threshold, picture_size, subject_box)
            generated = model.edit(reference, prompt, generation["seed"], generation["steps"], size=picture_size)
            scene = ImageOps.fit(generated.convert("RGB"), picture_size, Image.Resampling.LANCZOS)
            foreground = find_foreground(scene, models["mask"], threshold)
            if answer["edge"] == "blend":
                scene = blend_edges(scene, foreground, position=(artwork["x"], artwork["y"]), canvas_size=size)
            scene.save(result / artwork["file"])
            answer, layout, final_review_reason = review_finished(ai, answer, layout, scene, foreground, result,
                                                                  design_input, images[1:])
            Image.fromarray(foreground).save(result / "foreground.png")
        layout.update(fields=fields, brief=brief, design=answer, scene_prompt=prompt, preview=preview,
                      final_review_reason=final_review_reason)
        if preview:
            scene.save(result / "scene.png")
        poster = compose(layout, result)
        poster.save(result / "ad.png")
        if revision_info:
            layout["revision"] = revision_info
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
    drafts = parser.add_mutually_exclusive_group()
    drafts.add_argument("--designs", type=Path, help="상품별로 저장된 구도 시안을 그대로 사용하기")
    drafts.add_argument("--revise", type=Path, help="이 폴더의 완성본을 보고 사진부터 다시 생성해 비교하기")
    parser.add_argument("--rejected", type=Path, help="다음 개정에서 피해야 할 보류 후보 폴더")
    parser.add_argument("--low-vram", action="store_true", help="가중치를 나눠 올려 그래픽 메모리를 절약하기")
    args = parser.parse_args()
    if args.rejected and not args.revise:
        parser.error("--rejected에는 --revise가 필요합니다")
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
        previous = args.revise / folder.name if args.revise else None
        rejected = args.rejected / folder.name if args.rejected else None
        if rejected and not rejected.is_dir():
            rejected = None
        make_scene_ad(photo, item["fields"], item.get("idea", ""), models, folder,
                      references=references, preview=args.preview, proposal=proposal, previous=previous,
                      rejected=rejected)
        print(folder / "ad.png", flush=True)


if __name__ == "__main__":
    main()
