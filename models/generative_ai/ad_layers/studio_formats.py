"""채택한 광고의 사진을 보호하면서 게시 규격마다 글자를 다시 배치합니다."""

from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .creative_brief import ask_stage, required_text
from .layers import compose, load_layout, load_settings, save_layout, to_float
from .scene_ad import check_text_overlap, design_layers, rendered_text_bounds
from .studio_edits import foreground_mask, line_break_after
from .text_render import LINE_HEIGHT, minimum_font_size


FORMATS = {"1:1": (1024, 1024), "4:5": (864, 1080), "9:16": (1080, 1920), "16:9": (1280, 720),
           "thumbnail": (600, 600)}

POSITION = """Pick the placement of the same photograph for one poster in the new ratio.
The first image is the approved poster. The following previews are in position_order. Their extended
edges preview where space will be filled. Choose where the product, copy and negative space work together.
Check which placement gives the headline and supporting copy one useful quiet area near the product
and keeps the scene anchored. Avoid distributing space into two large unused strips. Prefer extending
an already simple part of the scene over extending furniture, hard edges, or important props.
Keep every visible product and retain the campaign's visual focus. Return JSON with position from
position_order and reason as one concrete Korean sentence about the intended composition."""

MARGIN_CHECK = """Review the newly filled margins of a reframed product photo. The first image is the actual
filled photograph; the second shows the same photograph with only stretched preview margins. photo_bounds
identifies the unmodified photo in whole canvas percentages, left, top, right, bottom. Only inspect regions
outside those bounds. Look at the corners as well as seams. Reject a new object, glow, smear, or broken
surface that draws the eye or interrupts a continuous edge. Allow a naturally quiet extension. Return
ready (boolean), side (one of margin_sides if rejected, null if ready), and observation (one concrete Korean
sentence naming what is visible and where)."""

REFRAME = """Reframe the approved poster's exact typography for a new aspect ratio. The first image is the
approved poster, the second is the actual photo in the new frame, and the third shows protected product
pixels tinted in red. The fourth, if present, is a rejected rendered poster. Keep the photo unchanged. Use
visible space outside the tinted shapes to place the exact headline, product name and price. Keep the approved
campaign's colors and hierarchy. Pick fonts from the supplied list. Use an integer from 2 to 20 for size,
as a percentage of the new canvas width. Group the product name and price while letting the headline lead.
Read the new photo before choosing positions. In a tall canvas, do not separate title and product with a wide
unused stretch of background. Place the nearest title line beside or near the feature in the actual framing.
In a wide canvas with no clear footer, keep the product name and price beside the headline or the product.
Return a JSON object with boxes and styles keyed by headline, product_name and price. Each box is
[left, top, width, height] in whole percentages of the poster. Keep them inside the poster and off one another.
Each style has font, align (left, center, or right), color (#RRGGBB), size, and max_lines (1 or 2,
always 1 for price). Each box must be at least min_box_height_percent for its line count and needs more height
for larger lettering. Allow enough width for any preferred_first_lines. For the headline only, optional
depth behind_subject lets a small part of the headline tuck behind the product. If the approved headline was
placed as two independent lines, optional styles.headline.lines has two objects with new boxes in whole poster
percentages inside the headline box. An object can override size, font, align, color or depth. Reposition the
lines for the new canvas or omit lines if one shared heading block works better. An optional accent is null or an object
with shape (rectangle or ellipse), box, color, and layer (overlay or behind_subject when the shape deliberately
meets the visible product silhouette). Use the previous_layout and layout_error
to repair failed boxes while keeping the parts that fit."""

FORMAT_READING = """Compare the approved poster and the actual reframed poster in the first and second images.
Judge the second at mobile size, using rendered_ink and product_bounds as evidence. Check the exact copy,
contrast, title line grouping, and how a viewer reaches the product and price. Quiet margins around the whole
composition are allowed. A large empty band between the title and subject, with no visible photographic cue
across it, breaks the reading path. A pair of title lines needs to read as one phrase before the viewer reaches
the product. Require enough contrast and space even on a wide crop. Return ready (boolean) and observation
(one concrete Korean sentence describing actual placement or what must change)."""

COPY_BREAK = """검토할 광고 문구를 두 줄에 놓을 때 자연스러운 첫 줄을 고릅니다.
문구를 그대로 읽고 문장의 절이나 호흡이 끝나는 곳에서 나눕니다. 상품을 설명하는 수량과 단위, 이름처럼
같이 읽어야 하는 묶음을 유지합니다. 첫 줄의 끝에 낱말을 외따로 남기지 마세요.
문구를 고쳐 쓰지 마세요. 입력에 있는 각 역할을 키로 두고, 원래 문구에서 그대로 떼어낸 첫 줄 또는
자연스러운 줄바꿈이 없다는 뜻의 null을 값으로 하는 JSON 객체를 반환하세요."""


def extend_edges(contained, size, placement):
    left, top = int((size[0] - contained.width) * placement), int((size[1] - contained.height) * placement)
    columns = (left, size[0] - contained.width - left)
    rows = (top, size[1] - contained.height - top)
    pixels = np.pad(np.asarray(contained), (rows, columns, (0, 0)), mode="edge")
    frame = Image.fromarray(pixels)
    radius = max(1, min(size) / 44)
    x, y = np.arange(size[0]), np.arange(size[1])
    horizontal = np.maximum(left - x, x - left - contained.width + 1)
    vertical = np.maximum(top - y, y - top - contained.height + 1)
    distance = np.maximum(horizontal[None, :], vertical[:, None])
    dissolve = Image.fromarray((255 * np.clip(distance / (radius * 2), 0, 1)).astype(np.uint8))
    frame = Image.composite(frame.filter(ImageFilter.GaussianBlur(radius)), frame, dissolve)
    frame.paste(contained, (left, top))
    return frame, (left, top, left + contained.width, top + contained.height)


def reflect_margin(filled, contained, placed, side):
    left, top, right, bottom = placed
    width, height = filled.size
    padding = ((top, height - bottom), (left, width - right), (0, 0))
    reflected = Image.fromarray(np.pad(np.asarray(contained), padding, mode="reflect"))
    bounds = {"left": (0, 0, left, height), "right": (right, 0, width, height),
              "top": (0, 0, width, top), "bottom": (0, bottom, width, height)}[side]
    result = filled.copy()
    result.paste(reflected.crop(bounds), bounds[:2])
    return result


def frame_photo(photo, mask, size, expand=None, choose_position=None, check_margin=None):
    width, height = photo.size
    target_width, target_height = size
    ratio = target_width / target_height
    crop_width, crop_height = min(width, round(height * ratio)), min(height, round(width / ratio))
    left, top, right, bottom = mask.getbbox()
    fits = right - left <= crop_width and bottom - top <= crop_height
    if not fits:
        padding = round(min(width, height) * 0.04)
        crop_width = min(width, max(crop_width, right - left + 2 * padding))
        crop_height = min(height, max(crop_height, bottom - top + 2 * padding))
    x = min(left, width - crop_width, max(0, right - crop_width, (width - crop_width) // 2))
    y = min(top, height - crop_height, max(0, bottom - crop_height, (height - crop_height) // 2))
    bounds = (x, y, x + crop_width, y + crop_height)
    photo, mask = photo.crop(bounds), mask.crop(bounds)
    if fits:
        return photo.resize(size, Image.Resampling.LANCZOS), mask.resize(size, Image.Resampling.NEAREST)
    contained = ImageOps.contain(photo, size, Image.Resampling.LANCZOS)
    extra_width, extra_height = size[0] - contained.width, size[1] - contained.height
    positions = ("left", "center", "right") if extra_width > extra_height else ("top", "center", "bottom")
    framing = "center"
    if choose_position is not None and max(extra_width, extra_height) >= min(size) * 0.06:
        choices = {name: extend_edges(contained, size, fraction)
                   for name, fraction in zip(positions, (0.15, 0.5, 0.85), strict=True)}
        framing = choose_position({name: item[0] for name, item in choices.items()})
        if not isinstance(framing, str) or framing not in choices:
            raise ValueError("게시 규격에 사용할 사진 위치를 고르지 못했습니다")
        frame, placed = choices[framing]
    else:
        frame, placed = extend_edges(contained, size, 0.5)
    for attempt in range(2):
        if expand is None or placed == (0, 0, *size):
            break
        margin = Image.new("L", size, 255)
        margin.paste(0, placed)
        filled = Image.composite(expand(frame, np.asarray(margin) != 0), frame, margin)
        side = check_margin(filled, frame, placed, "relocated" if attempt else "initial") if check_margin else None
        if side is None:
            frame = filled
            break
        if attempt:
            raise ValueError("게시 규격에서 사진의 새 가장자리를 확정하지 못했습니다")
        reflected = reflect_margin(filled, contained, placed, side)
        if check_margin(reflected, frame, placed, "reflected") is None:
            frame = reflected
            break
        fraction = 0 if side in ("left", "top") else 1
        frame, placed = extend_edges(contained, size, fraction)
    framed_mask = Image.new("L", size)
    left, top, right, bottom = placed
    framed_mask.paste(mask.resize((right - left, bottom - top), Image.Resampling.NEAREST), (left, top))
    return frame, framed_mask


def fill_margin(frame, margin):
    from ..photo_retouch.adjustments import to_image
    from ..photo_retouch.cleanup import inpaint, load_inpaint_model

    return to_image(inpaint(load_inpaint_model(), to_float(frame), margin, 1280))


def format_styles(ai, styles, layers, first_lines):
    if not isinstance(styles, dict):
        return styles
    requested = {
        layer["role"]: layer["text"] for layer in layers
        if layer["role"] != "price" and layer["role"] not in first_lines
        and isinstance(styles.get(layer["role"]), dict) and styles[layer["role"]].get("max_lines") == 2
    }
    if requested:
        answer = ask_stage(ai, "studio_copy_break", COPY_BREAK, requested)
        if answer.keys() != requested.keys():
            raise ValueError("문구마다 두 줄로 나눌 위치를 골라야 합니다")
        for role, first_line in answer.items():
            if first_line is not None:
                line_break_after(requested[role], first_line)
        first_lines.update(answer)
    revised = styles.copy()
    for layer in layers:
        role = layer["role"]
        if not isinstance(styles.get(role), dict):
            continue
        style = styles[role].copy()
        if "emphasis" in layer:
            style.setdefault("emphasis", layer["emphasis"])
        if role in first_lines:
            style.pop("break_after", None)
            if style.get("max_lines") == 2:
                if first_lines[role] is None:
                    style["max_lines"] = 1
                else:
                    style["break_after"] = line_break_after(layer["text"], first_lines[role])
        if "break_after" not in style:
            style.pop("lines", None)
        revised[role] = style
    return revised


def adapt_format(source, output, format_name, ai, expand=None):
    layout = load_layout(source)
    size = FORMATS[format_name]
    threshold_mask = foreground_mask(source, layout)
    mask = Image.fromarray(threshold_mask.astype(np.uint8) * 255)
    if not mask.getbbox():
        raise ValueError("크기 변경에 사용할 제품 영역이 없습니다")
    with Image.open(source / layout["artwork"]["file"]) as original:
        photo = original.convert("RGB")
    design = layout["design"]
    with Image.open(source / "ad.png") as poster:
        reference = to_float(poster)
    copy = {item["role"]: item["text"] for item in layout["texts"]}
    position_info = {}

    def choose_position(previews):
        pictures = [to_float(ImageOps.contain(image, (540, 540), Image.Resampling.LANCZOS))
                    for image in previews.values()]
        answer = ask_stage(ai, "studio_format_position", POSITION,
                           {"size": size, "copy": copy, "position_order": list(previews)}, [reference, *pictures])
        position_info.update(position=answer.get("position"), reason=required_text(answer, "reason", "사진 배치"))
        return position_info["position"]

    def check_margin(filled, preview, placed, method):
        left, top, right, bottom = placed
        margins = (left, top, size[0] - right, size[1] - bottom)
        sides = [side for side, amount in zip(("left", "top", "right", "bottom"), margins, strict=True) if amount]
        bounds = [round(100 * edge / length) for edge, length in zip(placed, size * 2, strict=True)]
        answer = ask_stage(ai, "studio_format_margin_check", MARGIN_CHECK,
                           {"photo_bounds": bounds, "margin_sides": sides}, [to_float(filled), to_float(preview)])
        observation = required_text(answer, "observation", "사진 가장자리 검토")
        if answer.get("ready") is True:
            position_info.update(margin_review=observation, margin_method=method)
            if method == "relocated":
                position_info.update(position=position_info["margin_side"],
                                     reason="채운 여백에 흔적이 보여 원본 사진을 그쪽 가장자리에 붙였습니다")
            return None
        side = answer.get("side")
        if answer.get("ready") is not False or side not in sides:
            raise ValueError("게시 규격에서 채운 사진 가장자리의 검토 위치를 확인하지 못했습니다")
        if method == "relocated":
            raise ValueError(f"게시 규격에서 사진의 새 가장자리를 확정하지 못했습니다. {observation}")
        if method == "initial":
            position_info.update(margin_side=side, margin_rejection=observation)
        return side

    framing, framing_mask = frame_photo(photo, mask, size, expand, choose_position, check_margin)
    foreground = np.asarray(framing_mask) > 128
    tint = Image.blend(framing, Image.new("RGB", size, "#d84458"), 0.6)
    protected = Image.composite(tint, framing, framing_mask)
    images = [reference, to_float(framing), to_float(protected)]
    fonts = load_settings()["fonts"]
    bounds = [round(100 * point / side) for point, side in zip(framing_mask.getbbox(), (*size, *size), strict=True)]
    inputs = {"size": size, "copy": copy, "fonts": list(fonts), "approved_design": design, "product_bounds": bounds}
    if position_info:
        inputs["framing"] = position_info
    line_height = minimum_font_size(size[0]) * LINE_HEIGHT
    inputs["min_box_height_percent"] = {"one_line": ceil(100 * line_height / size[1]),
                                        "two_lines": ceil(200 * line_height / size[1])}
    first_lines = {layer["role"]: " ".join(layer["text"].split()[:layer["break_after"]])
                   for layer in layout["texts"] if layer.get("break_after") is not None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary)
        for _ in range(2):
            answer = ask_stage(ai, "studio_reframe", REFRAME, inputs | {"preferred_first_lines": first_lines}, images)
            try:
                styles = format_styles(ai, answer.get("styles"), layout["texts"], first_lines)
                revised = design | {"artwork_box": [0, 0, 100, 100], "boxes": answer.get("boxes"),
                                    "styles": styles, "accent": answer.get("accent")}
                revised.pop("subject_box", None)
                candidate = design_layers(revised, copy, size, fonts, stage)
                check_text_overlap(foreground, candidate, stage)
            except ValueError as error:
                inputs["layout_error"] = str(error)
                inputs["previous_layout"] = answer
                continue
            framing.save(stage / candidate["artwork"]["file"])
            framing_mask.save(stage / "foreground.png")
            poster = compose(candidate, stage)
            phone = ImageOps.contain(poster, (360, 640), Image.Resampling.LANCZOS)
            review = ask_stage(ai, "studio_format_reading", FORMAT_READING,
                               {"copy": copy, "product_bounds": bounds, "rendered_ink": rendered_text_bounds(candidate, stage)},
                               [reference, to_float(phone)])
            reading = required_text(review, "observation", "게시 규격 검토")
            if type(review.get("ready")) is not bool:
                raise ValueError("게시 규격의 글자와 상품 배치 판정을 확인하지 못했습니다")
            if not review["ready"]:
                inputs.update(layout_error=reading, previous_layout=revised)
                images[3:] = [to_float(poster)]
                continue
            poster.save(stage / "ad.png")
            candidate.update(fields=layout["fields"], brief=layout["brief"], design=revised,
                             scene_prompt=layout["scene_prompt"], format=format_name, final_review_reason=reading)
            if "art_direction_plan" in layout:
                candidate["art_direction_plan"] = layout["art_direction_plan"]
            if position_info:
                candidate["framing"] = position_info
            save_layout(candidate, stage)
            stage.rename(output)
            return position_info
    raise ValueError(inputs["layout_error"])
