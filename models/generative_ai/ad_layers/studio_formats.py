"""채택한 광고의 사진을 보호하면서 게시 규격마다 글자를 다시 배치합니다."""

from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .creative_brief import ask_stage, required_text
from .format_layout import (format_measurements, format_type_hierarchy, measured_styles, outer_margins, position_options,
                            preview_placement)
from .layers import compose, load_layout, load_settings, save_layout, to_float
from .scene_ad import (check_text_overlap, design_layers, make_scene_ad, rendered_text_bounds, tighten_text_groups,
                       touches_photo_edge)
from .staging import check_staged_identity
from .studio_edits import foreground_mask, line_break_after, repair_staged_detail
from .text_render import LINE_HEIGHT, minimum_font_size


FORMATS = {"1:1": (1024, 1024), "4:5": (864, 1080), "9:16": (1080, 1920), "16:9": (1280, 720),
           "thumbnail": (600, 600)}

POSITION = """Pick the placement of the same photograph for one poster in the new ratio.
The first image is the approved poster. The following previews are in position_order. Their extended
edges preview where space will be filled. Choose where the product, copy and negative space work together.
Some previews show the actual approved words fitted into space that clears the product. Use placement_notes
to see which previews contain fitted type. Compare the entire layout. Avoid leaving a large unoccupied outer
band just to align the product to one end of the frame. Outer_margins measures the space beyond the union of
product and fitted words as a percentage of the canvas on each side. Look there for a visible purpose. If a
large flat strip is empty in one preview and another keeps the same reading path with less dead space, prefer
the more anchored layout even if the first preview has a simpler edge to expand.
Check which placement gives the headline and supporting copy one useful quiet area near the product
and keeps the scene anchored. Avoid distributing space into two large unused strips. Prefer extending
an already simple part of the scene over extending furniture, hard edges, or important props.
Keep every visible product and retain the campaign's visual focus. Return JSON with position from
position_order and reason as one concrete Korean sentence about the intended composition."""

MARGIN_CHECK = """Review the newly filled margins of a reframed product photo. The first image is the actual
filled photograph; the second shows the same photograph with only stretched preview margins. photo_bounds
identifies the unmodified photo in whole canvas percentages, left, top, right, bottom. Only inspect regions
outside those bounds. Subsequent images show each added side plus a narrow strip of adjacent original photo,
in margin_sides order. Compare the material across that join. Look at the corners as well as seams.
Reject a new object, glow, smear, or broken surface. Even a short strip can be wrong when wood grain or a
cast shadow suddenly turns into parallel streaks at the join. Check for flipped patterns or repeating grains.
Reject a surface that turns into a different material even when the transition is gradual. Allow a naturally
quiet extension. Return
ready (boolean), side (one of margin_sides if rejected, null if ready), and observation (one concrete Korean
sentence naming what is visible and where)."""

REFRAME = """Reframe the approved poster's exact typography for a new aspect ratio. The first image is the
approved poster, the second is the actual photo in the new frame, and the third shows protected product
pixels tinted in red. The fourth, if present, is a rejected rendered poster. Keep the photo unchanged. Use
visible space outside the tinted shapes to place the exact headline, product name and price. Keep the approved
campaign's colors and hierarchy. Pick fonts from the supplied list. Use an integer from 2 to 20 for size,
as a percentage of the new canvas width. Group the product name and price. Keep the price visually
secondary unless approved_design.focus is price.
Read the new photo before choosing positions. In a tall canvas, do not separate title and product with a wide
unused stretch of background. Place the nearest title line beside or near the feature in the actual framing.
In a wide canvas with no clear footer, keep the product name and price beside the headline or the product.
When intent is provided, address that visible problem on the new photo without rewriting copy.
In a same-size repair, preserve the approved type's reading order and legibility. When the focal feature is
centered, start by looking for a shared axis around it. Move the offer to a separate corner only when a visible
line, action or contrast connects it to the reading path. Keep the offer comfortably readable at phone size.
placement_options holds exact boxes checked against the product mask using the approved text sizes. If one
is available, start by choosing its key as placement. You may also return colors as a map from a text role
to a #RRGGBB when needed for contrast, including headline_line_1 or headline_line_2 for a split heading.
Keep the measured boxes and font_sizes when selecting an option. If the
available options do not fit after rendering, return a custom layout using boxes and styles instead.
On retry, look at why previous_placement or the rendered poster failed before choosing a placement again.
For a margin error, first shift the affected text group inward with its alignment intact. Keep the approved
alignment and size of the other copy unless the error explicitly names a broken reading path.
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
Judge the second at mobile size, using rendered_ink and product_bounds as evidence. The third image is the
same reframed poster at full size. Check whether a complete product has enough room at its nearest edge,
including its contact shadow. Inspect the background there for a seam or stretched detail. Accept a clearly
intentional detail crop when it serves the campaign. Check the exact copy,
contrast, title line grouping, and how a viewer reaches the product and price. Check what draws the first
glance. The price may lead when focus is price. Otherwise it should read clearly without taking attention
from the intended focus. Quiet margins around the whole composition are allowed.
The fourth image is the actual area surrounding the product name and price. Look there before approving.
Reject a strong product edge, highlight, shadow, or texture interrupting the quiet space immediately
around those words. A recognizable letter can still look attached to a rim or reflection. Inspect the
actual pixels even if the outline in product_bounds misses the visible detail. Suggest one clear placement
change when this area fails. Treat this as a type repair unless the photograph has no usable text area.
A large empty band between the title and subject, with no visible photographic cue
across it, breaks the reading path. A pair of title lines needs to read as one phrase before the viewer reaches
the product. Also use outer_margins to inspect the actual space outside the product and all the lettering.
An offer stranded apart from the headline and advertised detail is a layout defect even when every word is
legible by itself. Give the specific gap or missing visual connection in the observation.
If intent describes a same-size repair, check that the photographed feature is more legible and the
unused region visibly shrank in the second image.
Read the long axis from edge to edge. An enlarged stretch of featureless wall, floor or surface has no
photographic purpose merely because it balances a matching empty stretch at the other end. Check whether
the subject itself meaningfully fills the long axis and directs the eye. If the scene was originally arranged
along the other axis, can changing the text position actually fix it without separating the words from the
product or making the product smaller? Require enough contrast and space even on a wide crop.
Return ready (boolean), observation (one concrete Korean sentence), and rebuild_photo (boolean, true only when
ready is false and the photograph's arrangement or empty filled edges require a native composition)."""

NATIVE_REVIEW = """Review a poster newly photographed for one publishing ratio. Image 1 is the original product
photograph, image 2 is the approved poster, image 3 is the attempted resize when present, and the last
two images are the new finished poster and its phone preview.
If has_staged_group is true, the image immediately before the finished poster shows the planned group
made from real product pixels. Compare count, colors, materials and the new depth ordering with that guide.
Check product identity, including the exact visible versions, essential details, visible packaging and component
count. Image 2 defines the campaign motif and styling. Match that motif without merely framing the original
composition in empty bands. Look along the long axis for the specific visible arrangement or photographic cue
that solves format_issue and connects the subject, headline and offer. A change in background alone is insufficient.
Check the required copy, its prominence and contrast at phone size, and the natural scene and edges at full size.
Return ready (boolean) and observation (one concrete Korean sentence describing the visible improvement or defect)."""

COPY_BREAK = """검토할 광고 문구를 두 줄에 놓을 때 자연스러운 첫 줄을 고릅니다.
문구를 그대로 읽고 문장의 절이나 호흡이 끝나는 곳에서 나눕니다. 상품을 설명하는 수량과 단위, 이름처럼
같이 읽어야 하는 묶음을 유지합니다. 첫 줄의 끝에 낱말을 외따로 남기지 마세요.
문구를 고쳐 쓰지 마세요. 입력에 있는 각 역할을 키로 두고, 원래 문구에서 그대로 떼어낸 첫 줄 또는
자연스러운 줄바꿈이 없다는 뜻의 null을 값으로 하는 JSON 객체를 반환하세요."""


class FormatPhotoIssue(ValueError):
    def __init__(self, observation, poster=None):
        super().__init__(observation)
        self.poster = poster


def extend_edges(contained, size, placement, *, offset=None):
    left, top = (offset if offset is not None else
                 (int((size[0] - contained.width) * placement), int((size[1] - contained.height) * placement)))
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


def reflect_margin(filled, contained, product, placed, side):
    left, top, right, bottom = placed
    width, height = filled.size
    px, py, pr, pb = product.getbbox()
    clearance = max(1, min(contained.size) // 50)
    strips = {"left": (0, 0, max(0, px - clearance), contained.height),
              "right": (min(contained.width, pr + clearance), 0, contained.width, contained.height),
              "top": (0, 0, contained.width, max(0, py - clearance)),
              "bottom": (0, min(contained.height, pb + clearance), contained.width, contained.height)}
    sx, sy, ex, ey = strips[side]
    if sx == ex or sy == ey:
        return None
    bounds = {"left": (0, top, left, bottom), "right": (right, top, width, bottom),
              "top": (left, 0, right, top), "bottom": (left, bottom, right, height)}[side]
    amount = (left, width - right) if side in ("left", "right") else (top, height - bottom)
    before, after = (amount[0], 0) if side in ("left", "top") else (0, amount[1])
    padding = ((0, 0), (before, after), (0, 0)) if side in ("left", "right") else ((before, after), (0, 0), (0, 0))
    reflected = Image.fromarray(np.pad(np.asarray(contained.crop(strips[side])), padding, mode="reflect"))
    w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x, y = (0, 0) if before else (reflected.width - w, reflected.height - h)
    patch = reflected.crop((x, y, x + w, y + h))
    sw, sh = contained.size
    edges = {"left": (0, 0, 1, sh), "right": (sw - 1, 0, sw, sh),
             "top": (0, 0, sw, 1), "bottom": (0, sh - 1, sw, sh)}
    soft = contained.crop(edges[side]).resize((w, h)).filter(ImageFilter.GaussianBlur(min(contained.size) / 40))
    horizontal = side in ("left", "right")
    depth = w if horizontal else h
    distance = np.arange(1, depth + 1) if not before else np.arange(depth, 0, -1)
    fade = np.clip(distance / max(1, min(depth, min(contained.size) // 6)), 0, 1)
    fade = (255 * fade**2 * (3 - 2 * fade)).astype(np.uint8)
    fade_mask = Image.fromarray(fade[None, :] if horizontal else fade[:, None]).resize((w, h))
    patch = Image.composite(soft, patch, fade_mask)
    result = filled.copy()
    result.paste(patch, bounds[:2])
    return result


def fit_crop(photo, mask, box, size, keep_clearance, check_margin):
    width, height = size
    crop = mask.crop(box)
    left, top, right, bottom = crop.getbbox()
    gaps = (left * width / crop.width, (crop.width - right) * width / crop.width,
            top * height / crop.height, (crop.height - bottom) * height / crop.height)

    def original():
        return photo.resize(size, Image.Resampling.LANCZOS, box=box), crop.resize(size, Image.Resampling.NEAREST)

    clearance = round(min(size) * 0.06)
    if not keep_clearance or min(gaps) == 0 or min(gaps) >= clearance:
        return original()

    # The scale must leave room for both clearances as well as the visible extent at either end.
    scale = 1
    for length, start, end in ((width, *gaps[:2]), (height, *gaps[2:])):
        scale = min(scale, (length - 2 * clearance) / (length - start - end),
                    (length - clearance) / (length - min(start, end)))
    smaller = (max(1, int(width * scale)), max(1, int(height * scale)))
    subject = crop.resize(smaller, Image.Resampling.NEAREST)
    if (bounds := subject.getbbox()) is None:
        raise ValueError("게시 규격에서 상품 영역이 너무 작습니다")
    left, top, right, bottom = bounds
    spare_x, spare_y = width - smaller[0], height - smaller[1]
    x = min(spare_x, max(0, clearance - left, min(spare_x // 2, width - right - clearance)))
    y = min(spare_y, max(0, clearance - top, min(spare_y // 2, height - bottom - clearance)))
    dx, dy = crop.width / smaller[0], crop.height / smaller[1]
    source_space = (box[0], box[1], photo.width - box[2], photo.height - box[3])
    left, top, right, bottom = (min(gap, int(space / pixel)) for gap, space, pixel in
                               zip((x, y, spare_x - x, spare_y - y), source_space, (dx, dy, dx, dy)))
    original_box = (box[0] - left * dx, box[1] - top * dy, box[2] + right * dx, box[3] + bottom * dy)
    original_size = (smaller[0] + left + right, smaller[1] + top + bottom)
    origin = (x - left, y - top)
    known = photo.resize(original_size, Image.Resampling.LANCZOS, box=original_box)
    filled, placed = extend_edges(known, size, 0.5, offset=origin)
    if placed != (0, 0, *size) and check_margin is not None:
        known_subject = Image.new("L", original_size)
        known_subject.paste(subject, (left, top))
        preview = filled
        for side, remaining in zip(("left", "top", "right", "bottom"),
                                   (placed[0], placed[1], width - placed[2], height - placed[3])):
            if remaining:
                reflected = reflect_margin(filled, known, known_subject, placed, side)
                if reflected is not None:
                    filled = reflected
        if check_margin(filled, preview, placed, "clearance") is not None:
            return original()
    visible = Image.new("L", size)
    visible.paste(subject, (x, y))
    return filled, visible


def frame_photo(photo, mask, size, expand=None, choose_position=None, check_margin=None, *, keep_clearance=False):
    width, height = photo.size
    target_width, target_height = size
    ratio = target_width / target_height
    crop_width, crop_height = min(width, round(height * ratio)), min(height, round(width / ratio))
    left, top, right, bottom = mask.getbbox()
    padding = round(min(width, height) * 0.04)
    left, top = max(0, left - padding), max(0, top - padding)
    right, bottom = min(width, right + padding), min(height, bottom + padding)
    fits = right - left <= crop_width and bottom - top <= crop_height
    if not fits:
        crop_width = max(crop_width, right - left)
        crop_height = max(crop_height, bottom - top)
    x = min(left, width - crop_width, max(0, right - crop_width, (width - crop_width) // 2))
    y = min(top, height - crop_height, max(0, bottom - crop_height, (height - crop_height) // 2))
    bounds = (x, y, x + crop_width, y + crop_height)
    if fits:
        return fit_crop(photo, mask, bounds, size, keep_clearance, check_margin)
    photo, mask = photo.crop(bounds), mask.crop(bounds)
    contained = ImageOps.contain(photo, size, Image.Resampling.LANCZOS)
    product = mask.resize(contained.size, Image.Resampling.NEAREST)

    def place_mask(placed):
        framed = Image.new("L", size)
        framed.paste(product, placed[:2])
        return framed

    extra_width, extra_height = size[0] - contained.width, size[1] - contained.height
    positions = ("left", "center", "right") if extra_width > extra_height else ("top", "center", "bottom")
    pl, pt, pr, pb = mask.getbbox()
    touches_start = (extra_width > 0 and pl == 0) or (extra_height > 0 and pt == 0)
    touches_end = (extra_width > 0 and pr == mask.width) or (extra_height > 0 and pb == mask.height)
    if touches_start and touches_end:
        raise FormatPhotoIssue("잘린 상품이 양쪽 사진 가장자리에 닿아 있어 이 규격에 맞는 구도로 다시 촬영해야 합니다")
    framing = "center"
    if touches_start or touches_end:
        frame, placed = extend_edges(contained, size, 0 if touches_start else 1)
    elif choose_position is not None and max(extra_width, extra_height) >= min(size) * 0.06:
        choices = {name: extend_edges(contained, size, fraction)
                   for name, fraction in zip(positions, (0.15, 0.5, 0.85), strict=True)}
        framing = choose_position({name: item[0] for name, item in choices.items()},
                                  {name: place_mask(item[1]) for name, item in choices.items()})
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
            raise FormatPhotoIssue("게시 규격에서 사진의 새 가장자리를 확정하지 못했습니다")
        reflected = reflect_margin(filled, contained, product, placed, side)
        if reflected is not None and check_margin(reflected, frame, placed, "reflected") is None:
            frame = reflected
            break
        if touches_start or touches_end:
            raise FormatPhotoIssue("상품이 사진 가장자리에 닿아 있어 여백의 이음새를 피할 수 없습니다")
        fraction = 0 if side in ("left", "top") else 1
        frame, placed = extend_edges(contained, size, fraction)
    return frame, place_mask(placed)


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


def selected_placement(answer, options, design):
    name = answer.get("placement")
    if name is None:
        return answer
    if not isinstance(name, str) or name not in options:
        raise ValueError("계산한 게시 규격 배치 중 하나를 골라야 합니다")
    plan = options[name]
    colors = answer.get("colors", {})
    if not isinstance(colors, dict):
        raise ValueError("게시 규격 문구의 색상을 확인하지 못했습니다")
    return {"boxes": plan["boxes"], "styles": measured_styles(design, plan, colors), "accent": None}


def check_type_margin(size, ink):
    width, height = size
    clearance = min(size) / 40
    for role, (left, top, right, bottom) in ink.items():
        edges = {"왼쪽": left * width, "위쪽": top * height,
                 "오른쪽": (100 - right) * width, "아래쪽": (100 - bottom) * height}
        side = min(edges, key=edges.get)
        if edges[side] < 100 * clearance:
            shift = ceil(clearance - edges[side] / 100)
            raise ValueError(f"{role} 글자가 포스터 {side} 가장자리에 너무 가깝습니다. "
                             f"글자 묶음을 안쪽으로 {shift}픽셀 이상 옮겨 주세요")


def offer_detail(poster, ink):
    width, height = poster.size
    context = min(poster.size) // 12
    left, top, right, bottom = zip(ink["product_name"], ink["price"], strict=True)
    return poster.crop((max(0, round(min(left) * width / 100) - context),
                        max(0, round(min(top) * height / 100) - context),
                        min(width, round(max(right) * width / 100) + context),
                        min(height, round(max(bottom) * height / 100) + context)))


def move_copy_inside(design, ink, size):
    if design.get("accent"):
        return None
    offsets = []
    for axis, length in enumerate(size):
        gap = ceil(100 * min(size) / (40 * length))
        first = min(bounds[axis] for bounds in ink.values())
        last = max(bounds[axis + 2] for bounds in ink.values())
        minimum, maximum = gap - first, 100 - gap - last
        if minimum > maximum:
            return None
        offsets.append(max(minimum, min(0, maximum)))
    dx, dy = offsets
    if max(abs(dx), abs(dy)) > 3 or not any(offsets):
        return None

    def translated(box):
        x, y, width, height = box
        return [x + dx, y + dy, width, height]

    boxes = {role: translated(box) for role, box in design["boxes"].items()}
    if any(x < 0 or y < 0 or x + width > 100 or y + height > 100 for x, y, width, height in boxes.values()):
        return None
    styles = design["styles"]
    if lines := styles["headline"].get("lines"):
        styles = styles | {"headline": styles["headline"] | {
            "lines": [line | {"box": translated(line["box"])} for line in lines]}}
    return design | {"boxes": boxes, "styles": styles}


def adapt_format(source, output, format_name, ai, expand=None, *, intent=None):
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
    dimensions = format_measurements(layout, size)

    def available(mask):
        bounds = [round(100 * edge / length) for edge, length in zip(mask.getbbox(), size * 2, strict=True)]
        return bounds, position_options(layout, size, np.asarray(mask) > 128, bounds, dimensions)

    def choose_position(previews, masks):
        pictures = []
        notes = {}
        for name, frame in previews.items():
            bounds, options = available(masks[name])
            choice = next(iter(options), None)
            ink = []
            if choice:
                frame = preview_placement(frame, layout, source, options[choice])
                ink = [(x, y, x + w, y + h) for x, y, w, h in options[choice]["boxes"].values()]
            notes[name] = {"product_bounds": bounds, "placement": choice, "outer_margins": outer_margins(bounds, ink)}
            pictures.append(to_float(ImageOps.contain(frame, (540, 540), Image.Resampling.LANCZOS)))
        answer = ask_stage(ai, "studio_format_position", POSITION,
                           {"size": size, "copy": copy, "position_order": list(previews), "placement_notes": notes,
                            **({"intent": intent} if intent else {})},
                           [reference, *pictures])
        position_info.update(position=answer.get("position"), reason=required_text(answer, "reason", "사진 배치"))
        return position_info["position"]

    def check_margin(filled, preview, placed, method):
        left, top, right, bottom = placed
        margins = (left, top, size[0] - right, size[1] - bottom)
        sides = [side for side, amount in zip(("left", "top", "right", "bottom"), margins, strict=True) if amount]
        bounds = [round(100 * edge / length) for edge, length in zip(placed, size * 2, strict=True)]
        width, height = size
        context = min(size) // 10
        details = {"left": (0, 0, min(width, left + context), height),
                   "top": (0, 0, width, min(height, top + context)),
                   "right": (max(0, right - context), 0, width, height),
                   "bottom": (0, max(0, bottom - context), width, height)}
        answer = ask_stage(ai, "studio_format_margin_check", MARGIN_CHECK,
                           {"photo_bounds": bounds, "margin_sides": sides},
                           [to_float(filled), to_float(preview), *[to_float(filled.crop(details[side])) for side in sides]])
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
            raise FormatPhotoIssue(f"게시 규격에서 사진의 새 가장자리를 확정하지 못했습니다. {observation}")
        if method == "initial":
            position_info.update(margin_side=side, margin_rejection=observation)
        if method == "clearance":
            position_info.update(margin_side=side, margin_rejection=observation, margin_method="original_crop")
        return side

    framing, framing_mask = frame_photo(photo, mask, size, expand, choose_position, check_margin,
                                        keep_clearance=design["crop"] == "preserve")
    foreground = np.asarray(framing_mask) > 128
    tint = Image.blend(framing, Image.new("RGB", size, "#d84458"), 0.6)
    protected = Image.composite(tint, framing, framing_mask)
    images = [reference, to_float(framing), to_float(protected)]
    fonts = load_settings()["fonts"]
    bounds, placements = available(framing_mask)
    inputs = {"size": size, "copy": copy, "fonts": list(fonts), "approved_design": design, "product_bounds": bounds}
    if intent:
        inputs["intent"] = intent
    inputs["placement_options"] = placements
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

        def render_copy(design):
            candidate = design_layers(design, copy, size, fonts, stage)
            check_text_overlap(foreground, candidate, stage)
            tighten_text_groups(foreground, candidate, stage)
            ink = rendered_text_bounds(candidate, stage)
            if not {"product_name", "price"} <= ink.keys():
                raise ValueError("상품명과 가격의 글자가 화면에 보여야 합니다")
            return candidate, ink

        for _ in range(2):
            answer = ask_stage(ai, "studio_reframe", REFRAME, inputs | {"preferred_first_lines": first_lines}, images)
            if placements and "layout_error" not in inputs and answer.get("placement") is None:
                answer = {"placement": next(iter(placements))}
            try:
                proposed = selected_placement(answer, placements, design)
                styles = format_styles(ai, proposed.get("styles"), layout["texts"], first_lines)
                styles = format_type_hierarchy(styles, design["focus"])
                revised = design | {"artwork_box": [0, 0, 100, 100], "boxes": proposed.get("boxes"),
                                    "styles": styles, "accent": proposed.get("accent")}
                revised.pop("subject_box", None)
                candidate, ink = render_copy(revised)
                try:
                    check_type_margin(size, ink)
                except ValueError:
                    shifted = move_copy_inside(revised, ink, size) if intent else None
                    if shifted is None:
                        raise
                    candidate, ink = render_copy(shifted)
                    check_type_margin(size, ink)
                    revised = shifted
            except ValueError as error:
                inputs["layout_error"] = str(error)
                inputs["previous_layout"] = answer
                continue
            framing.save(stage / candidate["artwork"]["file"])
            framing_mask.save(stage / "foreground.png")
            poster = compose(candidate, stage)
            phone = ImageOps.contain(poster, (360, 640), Image.Resampling.LANCZOS)
            review = ask_stage(ai, "studio_format_reading", FORMAT_READING,
                               {"copy": copy, "focus": design["focus"], "product_bounds": bounds, "rendered_ink": ink,
                                "outer_margins": outer_margins(bounds, ink.values()),
                                **({"intent": intent} if intent else {})},
                               [reference, to_float(phone), to_float(poster), to_float(offer_detail(poster, ink))])
            reading = required_text(review, "observation", "게시 규격 검토")
            if type(review.get("ready")) is not bool:
                raise ValueError("게시 규격의 글자와 상품 배치 판정을 확인하지 못했습니다")
            if not review["ready"]:
                if review.get("rebuild_photo") is True:
                    raise FormatPhotoIssue(reading, poster)
                inputs.update(layout_error=reading, previous_layout=revised)
                inputs["previous_placement"] = answer.get("placement")
                images[3:] = [to_float(poster)]
                continue
            poster.save(stage / "ad.png")
            candidate.update(fields=layout["fields"], brief=layout["brief"], design=revised,
                             scene_prompt=layout["scene_prompt"], format=format_name, final_review_reason=reading)
            if answer.get("placement") is not None:
                candidate["text_placement"] = answer["placement"]
            if "art_direction_plan" in layout:
                candidate["art_direction_plan"] = layout["art_direction_plan"]
            if position_info:
                candidate["framing"] = position_info
            save_layout(candidate, stage)
            stage.rename(output)
            return position_info
    raise ValueError(inputs["layout_error"])


def restore_format_detail(folder, layout, ai, models):
    def assess():
        with Image.open(folder / layout["artwork"]["file"]) as scene:
            return check_staged_identity(ai, folder, layout, scene)

    match, observation = assess()
    if match:
        return None
    try:
        repair_staged_detail(folder, layout, {"instruction": observation}, lambda: models, ai)
    except ValueError as error:
        raise ValueError(f"게시 규격에서 강조한 상품이 원본과 다릅니다. {observation} {error}") from error
    match, observation = assess()
    if not match:
        raise ValueError(f"게시 규격에서 강조한 상품이 원본과 다릅니다. {observation}")
    compose(layout, folder).save(folder / "ad.png")
    return observation


def rebuild_format(source, output, format_name, photo, fields, idea, ai, models, references, issue):
    size = FORMATS[format_name]
    approved = load_layout(source)
    copy = {row["role"]: row["text"] for row in approved["texts"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        folder = Path(temporary) / "candidate"
        layout = make_scene_ad(photo, fields, idea, models | {"ai": ai}, folder, previous=source,
                               references=references, format_rebuild=(size, str(issue)))
        if tuple(layout["size"]) != size or {row["role"]: row["text"] for row in layout["texts"]} != copy:
            raise ValueError("새 구도에서 채택 시안의 규격 또는 문구가 달라졌습니다")
        if layout["design"]["crop"] == "preserve" and touches_photo_edge(foreground_mask(folder, layout)):
            raise ValueError("새 게시용 사진에서 전체를 보여 줄 상품이 사진 가장자리에 닿습니다")
        check_type_margin(size, rendered_text_bounds(layout, folder))
        revision = layout.get("revision", {})
        restored = None
        if revision.get("staged_piece_box") is not None:
            restored = restore_format_detail(folder, layout, ai, models)
        with Image.open(source / "ad.png") as reference, Image.open(folder / "ad.png") as poster:
            phone = ImageOps.contain(poster, (360, 640), Image.Resampling.LANCZOS)
            before = [to_float(issue.poster)] if issue.poster is not None else []
            details = []
            if revision.get("staged_group"):
                with Image.open(folder / "staging.png") as guide:
                    details = [to_float(guide)]
            review = ask_stage(ai, "studio_format_native_review", NATIVE_REVIEW,
                               {"size": size, "format_issue": str(issue), "copy": copy,
                                "has_attempted_resize": bool(before), "has_staged_group": bool(details)},
                               [to_float(photo), to_float(reference), *before, *details, to_float(poster), to_float(phone)])
        observation = required_text(review, "observation", "새 게시 규격 검토")
        if review.get("ready") is not True:
            raise ValueError(f"게시 규격의 새 구도를 확정하지 못했습니다. {observation}")
        framing = {"method": "native", "resize_issue": str(issue), "review": observation}
        if restored:
            framing["detail_restoration"] = restored
        save_layout(layout | {"format": format_name, "framing": framing}, folder)
        folder.rename(output)
    return framing
