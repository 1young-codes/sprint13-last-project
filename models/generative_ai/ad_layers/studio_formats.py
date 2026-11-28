"""채택한 광고의 사진을 보호하면서 게시 규격마다 글자를 다시 배치합니다."""

from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageColor, ImageFilter, ImageOps
from scipy.ndimage import distance_transform_edt

from ..photo_retouch.adjustments import luminance, srgb_to_linear
from .creative_brief import TEXT_ROLES, ask_stage, required_text, validate_boxes
from .format_layout import (format_measurements, format_type_hierarchy, measured_styles, outer_margins, position_options,
                            preview_placement)
from .layers import compose, load_layout, load_settings, save_layout, to_float
from .scene_ad import (action_mask, check_text_overlap, check_type_margin, design_layers, find_action_regions,
                       make_scene_ad, pixel_box, rendered_text_bounds, tighten_text_groups, touches_photo_edge)
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
position_order and reason as one concrete Korean sentence about the intended composition. When photo_cue is
provided, keep the complete physical action and the part of any hand or support making contact visible.
Keep context_feature recognizable when it is supplied, with enough space for the photographed evidence.
In a same-size repair, prefer a framing that makes the actual product more prominent and leaves coherent
space for the headline and offer near the advertised feature."""

CUE_CROP = """The first image is the actual photograph and the second is its current poster. Identify the
smallest rectangle containing all visible parts of the merchandise and the complete physical action in
photo_cue. Include the contact and support that make the action believable, such as fingers where they grip.
Exclude empty background or an incidental limb beyond its supporting contact. Preserve the actual product's
silhouette, label, and parts that already appear in the photograph. Locate a separate context feature when
the exact copy names a visibly present object or setting detail that makes the message concrete. Keep any
previously named context_feature if it is still visible. Localize the part of that feature that is actually
visible in the first image, including a clipped part at the very edge. Omit incidental props whose loss would
leave the message unchanged. Return JSON with protected_box for the merchandise and action, context_box
for the visible context feature, context_feature as a short description of the content of context_box, and
observation as one Korean sentence naming what must remain visible. Both rectangles use [left, top, width,
height] in percentages of the first image, or null when the corresponding region is absent."""

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
fits the requested composition, choose its key as placement. Look for a side position near the photographed
point of interest when the intent calls for text on that side. If a one-line title would push everything into
a top band, compare a natural two-line title and nearby offer in the side space. Use a custom layout when the
measured choices would lose that reading path. You may also return colors as a map from a text role
to a #RRGGBB when needed for contrast, including headline_line_1 or headline_line_2 for a split heading.
Keep the measured boxes and font_sizes when selecting an option. If the
available options do not fit after rendering, return a custom layout using boxes and styles instead.
On retry, look at why previous_placement or the rendered poster failed before choosing a placement again.
For a margin error, first shift the affected text group inward with its alignment intact. Keep the approved
alignment and size of the other copy unless the error explicitly names a broken reading path.
For a full layout, return a JSON object with boxes and styles keyed by headline, product_name and price. Each box is
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
The fifth image shows the area around the headline. The sixth shows the same area before adding type.
Compare them before approving. Report layout if a bright rim, highlight, or texture edge reaches the title
glyphs and makes them look attached to the product contour, even when every letter is readable. A visible,
purposeful intersection can work when headline_behind_subject is true and the entire line still reads cleanly.
A large empty band between the title and subject, with no visible photographic cue
across it, breaks the reading path. A pair of title lines needs to read as one phrase before the viewer reaches
the product. Also use outer_margins to inspect the actual space outside the product and all the lettering.
An offer stranded apart from the headline and advertised detail is a layout defect even when every word is
legible by itself. Give the specific gap or missing visual connection in the observation.
If intent describes a same-size repair, check that the photographed feature is more legible and the
unused region visibly shrank in the second image. Verify any requested relative position of the product and
copy in the rendered result. Report layout if the wording missed its requested position and the photo can stay.
For side-by-side copy and action, identify where the eye crosses from the words to the visible action. If all
copy hugs the top while the focal detail sits farther down and the side space beside it remains unused, ask
for placement closer to that detail when the image provides enough clear space.
If photo_cue is provided, find that physical relationship or effect in the reframed photo. Use photo when
the crop has removed it and a new composition is required to show it again.
If context_feature is provided, check that enough of that original scene detail remains recognizable and
still connects to the actual copy. Report photo if it vanished or is reduced to an unrecognizable sliver.
Read the long axis from edge to edge. An enlarged stretch of featureless wall, floor or surface has no
photographic purpose merely because it balances a matching empty stretch at the other end. Check whether
the subject itself meaningfully fills the long axis and directs the eye. If the scene was originally arranged
along the other axis, can changing the text position actually fix it without separating the words from the
product or making the product smaller? Require enough contrast and space even on a wide crop.
Return JSON with observation (one concrete Korean sentence), finding (one of ready, offer, layout, photo),
and headline_touches_product (boolean). Set headline_touches_product true when a visible product rim or
highlight reaches the headline glyphs, even if it is easy to read or the overlap is intentional.
Use offer if moving the name and price together can fix the only visible problem. If your observation asks
to move those two lines into free space and nothing else needs changing, this is an offer finding.
Use layout when the title, type size or multiple groups actually need revision. Use photo only when the
photograph's own arrangement or filled edges require a new composition."""

OFFER_POSITION = """The first image is the rejected poster. The remaining images are actual candidate posters,
in position_order, with only the name and price moved together. Each image shows the whole poster above,
and an enlarged view of the name and price area below. Choose the candidate that resolves the noted
conflict and keeps a clear reading path between the headline, product and offer. Look for real space around
the entire offer, including adjacent reflections and container edges. Avoid fixing one side by running the words
into a different edge. Judge the whole poster as well as the nearest photographic detail. Return JSON with
position (one key from position_order, or null if none is better) and observation (a concrete Korean sentence)."""

COPY_POSITION = """These are actual posters using the same photo, words and lettering. Only the position of
the complete copy group changes. The first poster is current. The others follow position_order. Compare at
poster size. Name the visible subject detail that should receive the viewer's eye after reading the copy.
Prefer a change only when the words and that detail form a clearer reading path. For motion or a visible
effect, look at the actual point of contact or effect, not merely the outline of the largest object.
Leave quiet breathing room around the words and avoid placing them next to hard rims, highlights or shadows.
Honor the requested side, alignment and hierarchy. Keep current if the movement only fills empty space,
breaks an existing alignment or introduces a new conflict. Return position (a key from position_order) and
observation (one Korean sentence naming the visible relationship that justifies the choice)."""

STACKED_COPY_POSITION = """These are actual posters using the same photo, words and lettering. Only the position of
the complete copy group changes. The cards follow position_order. Each card shows
the whole poster above and the exact same area of the poster enlarged below. Use the lower views to compare
the background between the lettering and the actual photographic edge, including any rim, highlight or shadow.
Name the visible subject detail that should receive the viewer's eye after reading the copy.
Prefer a change only when the words and that detail form a clearer reading path. For motion or a visible
effect, look at the actual point of contact or effect, not merely the outline of the largest object.
For stacked copy and subject, look at the blank strip between the last line and the subject and the outer
margin above the complete group. If the strip splits the poster into disconnected parts, prefer the candidate
that reduces it while leaving a clear gap to the visible edge. Reject candidates where the edge crowds the letters.
Honor the requested side, alignment and hierarchy. Return position (a key from position_order, or null if
there is no meaningful visual difference) and observation (one Korean sentence comparing the gap or visual
relationship to the alternatives).
First compare the enlarged gaps and rule out layouts where the final row appears
attached to the visible rim or bright object. Among the remaining aligned options, prefer the one whose
shorter clear interval joins the copy and subject without crowding. Name both the usable shortened interval
and the crowding rejected in the alternative. Compare any large vacant band to the visible gap between
the text lines."""

NATIVE_REVIEW = """Review a poster newly photographed for one publishing ratio. Image 1 is the original product
photograph, image 2 is the approved poster, image 3 is the attempted resize when present, and the last
two images are the new finished poster and its phone preview.
If has_staged_group is true, the image immediately before the finished poster shows the planned group
made from real product pixels. Compare count, colors, materials and the new depth ordering with that guide.
Check product identity, including the exact visible versions, essential details, visible packaging and component
count. Image 2 defines the campaign motif and styling. Match that motif without merely framing the original
composition in empty bands. Look along the long axis for the specific visible arrangement or photographic cue
that solves format_issue and connects the subject, headline and offer. A change in background alone is insufficient.
When photo_cue is supplied, locate it in the final composition and check the actual direction and position of
the elements. Both parts can be visible while their planned relationship is absent.
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


def reflect_margins(frame, photo, content, placed):
    left, top, right, bottom = placed
    gaps = (left, top, frame.width - right, frame.height - bottom)
    for side, amount in zip(("left", "top", "right", "bottom"), gaps, strict=True):
        if amount and (filled := reflect_margin(frame, photo, content, placed, side)) is not None:
            frame = filled
    return frame


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
        filled = reflect_margins(filled, known, known_subject, placed)
        if check_margin(filled, preview, placed, "clearance") is not None:
            return original()
    visible = Image.new("L", size)
    visible.paste(subject, (x, y))
    return filled, visible


def frame_photo(photo, mask, size, expand=None, choose_position=None, check_margin=None, *, keep_clearance=False,
                protected_box=None):
    width, height = photo.size
    target_width, target_height = size
    ratio = target_width / target_height
    crop_width, crop_height = min(width, round(height * ratio)), min(height, round(width / ratio))
    left, top, right, bottom = protected_box or mask.getbbox()
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
        if protected_box is not None and choose_position is not None:
            min_x, max_x = max(0, right - crop_width), min(left, width - crop_width)
            min_y, max_y = max(0, bottom - crop_height), min(top, height - crop_height)
            if max(max_x - min_x, max_y - min_y) >= min(width, height) * 0.06:
                origins = {"start": (min_x, min_y), "center": (x, y), "end": (max_x, max_y)}
                seen = set()
                choices = {}
                for name, (px, py) in origins.items():
                    if (px, py) not in seen:
                        crop = (px, py, px + crop_width, py + crop_height)
                        choices[name] = fit_crop(photo, mask, crop, size, keep_clearance, check_margin)
                        seen.add((px, py))
                selected = choose_position({name: pair[0] for name, pair in choices.items()},
                                           {name: pair[1] for name, pair in choices.items()})
                if not isinstance(selected, str) or selected not in choices:
                    raise ValueError("촬영 연출을 유지할 사진 위치를 고르지 못했습니다")
                return choices[selected]
        return fit_crop(photo, mask, bounds, size, keep_clearance, check_margin)
    photo, mask = photo.crop(bounds), mask.crop(bounds)
    contained = ImageOps.contain(photo, size, Image.Resampling.LANCZOS)
    product = mask.resize(contained.size, Image.Resampling.NEAREST)
    if not product.getbbox():
        raise FormatPhotoIssue("게시 규격에서 상품 영역이 너무 작습니다")
    margin_content = product
    if protected_box is not None:
        margin_content = mask.copy()
        px, py, pr, pb = protected_box
        margin_content.paste(255, (px - x, py - y, pr - x, pb - y))
        margin_content = margin_content.resize(contained.size, Image.Resampling.NEAREST)

    def place_mask(placed):
        framed = Image.new("L", size)
        framed.paste(product, placed[:2])
        return framed

    extra_width, extra_height = size[0] - contained.width, size[1] - contained.height
    positions = ("left", "center", "right") if extra_width > extra_height else ("top", "center", "bottom")
    pl, pt, pr, pb = margin_content.getbbox()
    touches_start = (extra_width > 0 and pl == 0) or (extra_height > 0 and pt == 0)
    touches_end = (extra_width > 0 and pr == contained.width) or (extra_height > 0 and pb == contained.height)
    if touches_start and touches_end:
        raise FormatPhotoIssue("보존할 피사체가 양쪽 사진 가장자리에 닿아 있어 이 규격에 맞는 구도로 다시 촬영해야 합니다")
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
    if expand is None and protected_box is not None and placed != (0, 0, *size):
        filled = reflect_margins(frame, contained, margin_content, placed)
        if check_margin and check_margin(filled, frame, placed, "reflected") is not None:
            raise FormatPhotoIssue("보존할 피사체는 남겼지만 사진 가장자리가 부자연스럽습니다", filled)
        return filled, place_mask(placed)
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
        reflected = reflect_margin(filled, contained, margin_content, placed, side)
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


def round_box_edges(box):
    if (not isinstance(box, list) or len(box) != 4
            or any(type(value) not in (int, float) or not 0 <= value <= 100 for value in box)):
        return box
    x, y, w, h = box
    if x + w > 100 or y + h > 100:
        return box
    return [round(x), round(y), round(x + w) - round(x), round(y + h) - round(y)]


def protected_photo_bounds(plan, mask, keep_clearance, context_feature):
    def region(name):
        try:
            box = validate_boxes({name: round_box_edges(plan.get(name))}, (name,))[name]
        except ValueError:
            return None
        x, y, w, h = pixel_box(box, mask.size)
        return x, y, x + w, y + h

    merchandise = region("protected_box")
    if merchandise and not mask.crop(merchandise).getbbox():
        merchandise = None
    context = region("context_box") if context_feature else None
    if keep_clearance and not context_feature:
        return None
    if not merchandise and not context:
        return None
    boxes = [box for box in (merchandise, context) if box is not None]
    if keep_clearance or merchandise is None:
        boxes.append(mask.getbbox())
    left, top, right, bottom = zip(*boxes, strict=True)
    return min(left), min(top), max(right), max(bottom)


def selected_placement(answer, options, design):
    name = answer.get("placement")
    if name is None:
        boxes, styles = answer.get("boxes"), answer.get("styles")
        if isinstance(boxes, dict):
            answer = answer | {"boxes": {role: round_box_edges(box) for role, box in boxes.items()}}
        if isinstance(styles, dict) and isinstance(heading := styles.get("headline"), dict):
            if isinstance(lines := heading.get("lines"), list):
                lines = [line | {"box": round_box_edges(line.get("box"))} if isinstance(line, dict) else line
                         for line in lines]
                answer = answer | {"styles": styles | {"headline": heading | {"lines": lines}}}
        return answer
    if not isinstance(name, str) or name not in options:
        raise ValueError("계산한 게시 규격 배치 중 하나를 골라야 합니다")
    plan = options[name]
    colors = answer.get("colors", {})
    if not isinstance(colors, dict):
        raise ValueError("게시 규격 문구의 색상을 확인하지 못했습니다")
    return {"boxes": plan["boxes"], "styles": measured_styles(design, plan, colors), "accent": None}


def text_detail(poster, bounds):
    width, height = poster.size
    context = min(poster.size) // 12
    left, top, right, bottom = zip(*bounds, strict=True)
    return poster.crop((max(0, round(min(left) * width / 100) - context),
                        max(0, round(min(top) * height / 100) - context),
                        min(width, round(max(right) * width / 100) + context),
                        min(height, round(max(bottom) * height / 100) + context)))


def offer_detail(poster, ink):
    return text_detail(poster, (ink["product_name"], ink["price"]))


def check_offer_spacing(ink, size):
    first, second = sorted((ink["product_name"], ink["price"]))
    if min(first[3], second[3]) <= max(first[1], second[1]):
        return
    width, height = size
    gap = (second[0] - first[2]) * width
    letter_height = max(first[3] - first[1], second[3] - second[1]) * height
    if gap > 3 * letter_height:
        raise ValueError("상품명과 가격 사이의 빈 공간이 너무 넓습니다. 제목과 사진은 두고 두 문구를 가까이 배치하세요")


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
    return translate_copy(design, dx, dy)


def translate_copy(design, dx, dy):
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


def stacked_copy_direction(design, ink, foreground):
    if design["styles"]["headline"]["align"] != "center":
        return None
    height, width = foreground.shape
    rows = np.flatnonzero(foreground.any(axis=1))
    columns = np.flatnonzero(foreground.any(axis=0))
    if not rows.size:
        return None
    product_center = 50 * (columns[0] + columns[-1]) / width
    headline_center = sum(ink["headline"][::2]) / 2
    if abs(headline_center - product_center) > 5:
        return None
    if max(bounds[3] for bounds in ink.values()) < 100 * rows[0] / height:
        return 1
    if min(bounds[1] for bounds in ink.values()) > 100 * rows[-1] / height:
        return -1
    return None


def copy_position_options(design, layout, ink, foreground, stage):
    if design.get("accent") or any(text.get("depth") == "behind_subject" or
                                   any(line.get("depth") == "behind_subject" for line in text.get("lines", ()))
                                   for text in layout["texts"]):
        return {}
    size = layout["size"]
    short_edge = min(size)
    keepout = distance_transform_edt(~foreground) < short_edge / 80
    toward_product = stacked_copy_direction(design, ink, foreground)
    choices = {}
    rooms = []
    for axis, side in enumerate(size):
        first = min(bounds[axis] for bounds in ink.values())
        last = max(bounds[axis + 2] for bounds in ink.values())
        margin = ceil(100 * short_edge / (40 * side))
        rooms.append((max(0, first - margin), max(0, 100 - margin - last)))

    def reposition(layer, box):
        x, y, _, _ = pixel_box(box, size)
        return layer | {"x": x, "y": y}

    def translated_candidate(offset):
        revised = translate_copy(design, *offset)
        if revised is None:
            return None
        texts = []
        for text in layout["texts"]:
            role = text["role"]
            moved = reposition(text, revised["boxes"][role])
            if "lines" in text:
                rows = revised["styles"][role]["lines"]
                moved["lines"] = [reposition(line, row["box"])
                                  for line, row in zip(text["lines"], rows, strict=True)]
            texts.append(moved)
        candidate = layout | {"texts": texts}
        try:
            check_text_overlap(keepout, candidate, stage)
            revised_ink = rendered_text_bounds(candidate, stage)
            check_type_margin(size, revised_ink)
        except ValueError:
            return None
        return revised, candidate, revised_ink

    for axis in (1,) if toward_product else (0, 1):
        side = size[axis]
        cross_axis = 1 - axis
        nudge = round(5 * short_edge / size[cross_axis])
        left, right = (min(room, nudge) for room in rooms[cross_axis])
        sidesteps = (0,) if toward_product else dict.fromkeys((0, -left, right))
        for direction, room in zip((-1, 1), rooms[axis], strict=True):
            for distance in sorted({min(room, round(ratio * short_edge / side)) for ratio in (6, 12, 20, 30)}):
                if distance * side < 6 * short_edge:
                    continue
                offset = [0, 0]
                offset[axis] = distance * direction
                for step in sidesteps:
                    offset[cross_axis] = step
                    if result := translated_candidate(offset):
                        choices[f"position_{len(choices) + 1}"] = result
                        break
    if toward_product:
        title_y = design["boxes"]["headline"][1]
        if not any(toward_product * (revised["boxes"]["headline"][1] - title_y) > 0
                   for revised, _, _ in choices.values()):
            return {}
    return choices


def move_offer(design, offset):
    if (not isinstance(offset, list) or len(offset) != 2
            or any(type(value) is not int or abs(value) > 100 for value in offset) or not any(offset)):
        raise ValueError("상품명과 가격을 함께 옮길 위치가 필요합니다")
    dx, dy = offset
    steps = max(abs(dx), abs(dy))
    for step in range(steps, 0, -1):
        x_shift, y_shift = round(dx * step / steps), round(dy * step / steps)
        boxes = design["boxes"].copy()
        for role in ("product_name", "price"):
            x, y, w, h = boxes[role]
            boxes[role] = [x + x_shift, y + y_shift, w, h]
        try:
            validate_boxes(boxes, TEXT_ROLES)
        except ValueError:
            continue
        return design | {"boxes": boxes}
    raise ValueError("제목과 겹치지 않는 방향으로 상품명과 가격을 함께 옮겨 주세요")


def offer_position_options(design, layout, foreground, stage, tried):
    roles = ("product_name", "price")
    current_ink = rendered_text_bounds(layout, stage)
    left = min(current_ink[role][0] for role in roles)
    right = max(current_ink[role][2] for role in roles)
    headline = current_ink["headline"]
    align = design["styles"]["headline"]["align"]
    anchor = {"left": headline[0] - left, "center": round((headline[0] + headline[2] - left - right) / 2),
              "right": headline[2] - right}[align]
    _, title_top, _, title_height = design["boxes"]["headline"]
    top = min(design["boxes"][role][1] for role in roles)
    bottom = max(design["boxes"][role][1] + design["boxes"][role][3] for role in roles)
    directions = ((0, -7), (-2, -4), (2, -4), (-2, 0), (2, 0), (0, 7),
                  (anchor, title_top - bottom - 1), (anchor, title_top + title_height - top + 1),
                  (-6, -7), (6, -7))
    limits = []
    for axis, length in enumerate(layout["size"]):
        margin = ceil(100 * min(layout["size"]) / (40 * length))
        first = min(current_ink[role][axis] for role in roles)
        last = max(current_ink[role][axis + 2] for role in roles)
        limits.append((min(0, margin - first), max(0, 100 - margin - last)))
    seen = set(tried)
    choices = {}
    for direction in directions:
        offset = [max(low, min(value, high)) for value, (low, high) in zip(direction, limits, strict=True)]
        try:
            revised = move_offer(design, offset)
        except ValueError:
            continue
        key = tuple(revised["boxes"][role][axis] for role in roles for axis in (0, 1))
        if key in seen:
            continue
        seen.add(key)
        texts = []
        for text in layout["texts"]:
            role = text["role"]
            if role in roles:
                x, y, _, _ = pixel_box(revised["boxes"][role], layout["size"])
                text = text | {"x": x, "y": y}
            texts.append(text)
        candidate = layout | {"texts": texts}
        try:
            check_text_overlap(foreground, candidate, stage)
            ink = rendered_text_bounds(candidate, stage)
            check_type_margin(layout["size"], ink)
        except ValueError:
            continue
        choices[f"position_{len(choices) + 1}"] = (revised, candidate, ink, key)
    return choices


def paired_preview(poster, detail):
    preview = Image.new("RGB", (640, 640), "#f0efec")
    for source, top, height in ((poster, 0, 444), (detail, 452, 188)):
        image = ImageOps.contain(source, (640, height), Image.Resampling.LANCZOS)
        preview.paste(image, ((640 - image.width) // 2, top + (height - image.height) // 2))
    return preview


def offer_preview(poster, ink):
    return paired_preview(poster, offer_detail(poster, ink))


def ink_contrast(lightness, background):
    return np.quantile(np.maximum(lightness, background) / np.minimum(lightness, background), 0.1)


def ink_background(backdrop, text, ink):
    x, y, w, h = (text[key] for key in ("x", "y", "w", "h"))
    pixels = to_float(backdrop.crop((x, y, x + w, y + h)))[ink]
    return luminance(srgb_to_linear(pixels)) + 0.05


def color_lightness(rgb):
    return luminance(srgb_to_linear(np.asarray(rgb) / 255)) + 0.05


def legible_tone(color, background, minimum):
    original = np.asarray(ImageColor.getrgb(color))
    if not background.size or ink_contrast(color_lightness(original), background) >= minimum:
        return color
    for step in range(1, 21):
        for neutral in (255, 0):
            shade = tuple(np.rint(original + (neutral - original) * step / 20).astype(int))
            if ink_contrast(color_lightness(shade), background) >= minimum:
                return "#%02X%02X%02X" % shade
    return color


def recolor_ink(path, alpha, color):
    bitmap = Image.new("RGBA", alpha.size, color)
    bitmap.putalpha(alpha)
    bitmap.save(path)


def readable_headline_colors(design, layout, backdrop, folder, foreground):
    title = next(text for text in layout["texts"] if text["role"] == "headline")
    style = design["styles"]["headline"]
    original = title.get("lines", [title])
    marked_lines = sum(bool(line.get("emphasis")) for line in original)
    marked_color = None
    preview_scale = min(1, 360 / backdrop.width, 640 / backdrop.height)
    lines = []
    for line in original:
        # A mixed-color line is still judged in the rendered review.
        emphasis = line.get("emphasis")
        if emphasis and (marked_lines != 1 or emphasis["phrase"] != line["text"]):
            lines.append(line)
            continue
        path = folder / line["file"]
        with Image.open(path) as bitmap:
            alpha = bitmap.getchannel("A")
        ink = np.asarray(alpha) >= 128
        if line.get("depth") == "behind_subject":
            x, y = line["x"], line["y"]
            ink &= ~foreground[y:y + line["h"], x:x + line["w"]]
        bounds = alpha.getbbox()
        large = bounds and line["max_lines"] == 1 and (bounds[3] - bounds[1]) * preview_scale >= 18
        before = emphasis["color"] if emphasis else line["color"]
        color = legible_tone(before, ink_background(backdrop, line, ink), 3 if large else 4.5)
        if color != before:
            recolor_ink(path, alpha, color)
            line = line | {"color": color}
            if emphasis:
                marked_color = color
                line["emphasis"] = emphasis | {"color": color}
        lines.append(line)
    if lines == original:
        return design, layout
    if "lines" in title:
        title = title | {"lines": lines}
        style = style | {"lines": [row | {"color": line["color"]}
                                   for row, line in zip(style["lines"], lines, strict=True)]}
    else:
        title = lines[0]
        style = style | {"color": title["color"]}
    if marked_color:
        title = title | {"emphasis": title["emphasis"] | {"color": marked_color}}
        style = style | {"emphasis": style["emphasis"] | {"color": marked_color}}
    return design | {"styles": design["styles"] | {"headline": style}}, layout | {
        "texts": [title if text["role"] == "headline" else text for text in layout["texts"]]}


def readable_offer_colors(design, layout, backdrop, folder):
    roles = ("product_name", "price")
    headline = design["styles"]["headline"]
    palette = [headline["color"], *(line.get("color", headline["color"]) for line in headline.get("lines") or [])]
    offer = [text for text in layout["texts"] if text["role"] in roles]
    colors = list(dict.fromkeys([*(text["color"] for text in offer), *palette, "#000000", "#FFFFFF"]))
    lightness = {color: color_lightness(ImageColor.getrgb(color)) for color in colors}
    masks, contrasts = {}, {}
    for text in offer:
        with Image.open(folder / text["file"]) as layer:
            alpha = layer.getchannel("A")
        background = ink_background(backdrop, text, np.asarray(alpha) >= 128)
        if background.size:
            masks[text["role"]] = alpha
            contrasts[text["role"]] = {
                color: ink_contrast(value, background) for color, value in lightness.items()
            }
    minimum = 4.5
    low = {text["role"] for text in offer if text["role"] in contrasts
           and contrasts[text["role"]][text["color"]] < minimum}
    if not low:
        return design, layout

    common = next((color for color in colors if all(scores[color] >= minimum for scores in contrasts.values())), None)
    styles = design["styles"].copy()
    texts = []
    for text in layout["texts"]:
        role = text["role"]
        current = text["color"]
        if role in contrasts and (common is not None or role in low):
            scores = contrasts[role]
            color = common or next((color for color in palette if scores[color] >= minimum), None)
            if color is None:
                color = max((current, "#000000", "#FFFFFF"), key=scores.__getitem__)
            if color != current:
                recolor_ink(folder / text["file"], masks[role], color)
                styles[role] = styles[role] | {"color": color}
                text = text | {"color": color}
        texts.append(text)
    return design | {"styles": styles}, layout | {"texts": texts}


def copy_position_details(inks, foreground):
    height, width = foreground.shape
    size = (width, height)
    distances, nearest = distance_transform_edt(~foreground, return_indices=True)
    details = []
    for ink in inks:
        nearest_edge = None
        for role in TEXT_ROLES:
            left, top, right, bottom = ink[role]
            x, y, w, h = pixel_box([left, top, right - left, bottom - top], size)
            region = distances[y:y + max(1, h), x:x + max(1, w)]
            row, column = np.unravel_index(region.argmin(), region.shape)
            if nearest_edge is None or region[row, column] < nearest_edge[0]:
                near_y, near_x = nearest[:, y + row, x + column]
                edge_x, edge_y = 100 * near_x / width, 100 * near_y / height
                nearest_edge = (region[row, column], role, (edge_x, edge_y, edge_x, edge_y))
        _, role, edge = nearest_edge
        details.append(edge)
        details.extend((ink["product_name"], ink["price"]) if role != "headline" else (ink[role],))
    return details


def compare_copy_positions(ai, options, original, foreground, stage, inputs):
    design, _, ink = original
    if stacked_copy_direction(design, ink, foreground):
        alternatives = list(options.values())
        alternatives.insert(len(alternatives) // 2, original)
        choices = {f"layout_{index + 1}": item for index, item in enumerate(alternatives)}
        detail = copy_position_details([item[2] for item in alternatives], foreground)
        posters = [compose(item[1], stage) for item in alternatives]
        previews = [paired_preview(poster, text_detail(poster, detail)) for poster in posters]
        prompt = STACKED_COPY_POSITION
    else:
        choices = {"current": original, **options}
        previews = [ImageOps.contain(compose(item[1], stage), (640, 640), Image.Resampling.LANCZOS)
                    for item in choices.values()]
        prompt = COPY_POSITION
    choice = ask_stage(ai, "studio_format_copy_position", prompt, inputs | {"position_order": list(choices)},
                       [to_float(preview) for preview in previews])
    position = choice.get("position")
    if isinstance(position, str) and position in choices and choices[position] is not original:
        return choices[position], required_text(choice, "observation", "문구 위치 비교")
    return original, None


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
    photo_cue = layout.get("photo_cue")
    protected_box = None
    context_feature = layout.get("context_feature")
    keep_clearance = design["crop"] == "preserve"
    needs_crop = photo.width * size[1] != photo.height * size[0]
    if needs_crop and photo_cue:
        plan = ask_stage(ai, "studio_format_cue_crop", CUE_CROP,
                         {"photo_cue": photo_cue, "copy": copy, "intent": intent, "size": size,
                          "context_feature": context_feature},
                         [to_float(photo), reference])
        shared = plan.get("context_feature")
        if isinstance(shared, str) and shared.strip():
            context_feature = shared.strip()
        protected_box = protected_photo_bounds(plan, mask, keep_clearance, context_feature)

    def available(mask, regions=()):
        bounds = [round(100 * edge / length) for edge, length in zip(mask.getbbox(), size * 2, strict=True)]
        occupied = np.asarray(mask) > 128
        if regions:
            occupied |= action_mask(regions, size)
        return bounds, position_options(layout, size, occupied, bounds, dimensions)

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
                            **({"photo_cue": photo_cue} if photo_cue else {}),
                            **({"context_feature": context_feature} if context_feature else {}),
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
                                        keep_clearance=keep_clearance, protected_box=protected_box)
    foreground = np.asarray(framing_mask) > 128
    regions = find_action_regions(ai, framing, photo_cue) if photo_cue and layout.get("action_regions") != [] else []
    tint = Image.blend(framing, Image.new("RGB", size, "#d84458"), 0.6)
    occupied = Image.fromarray(foreground | action_mask(regions, size)) if regions else framing_mask
    protected = Image.composite(tint, framing, occupied)
    images = [reference, to_float(framing), to_float(protected)]
    fonts = load_settings()["fonts"]
    bounds, placements = available(framing_mask, regions)
    inputs = {"size": size, "copy": copy, "fonts": list(fonts), "approved_design": design, "product_bounds": bounds}
    if intent:
        inputs["intent"] = intent
    inputs["placement_options"] = placements
    if photo_cue:
        inputs["photo_cue"] = photo_cue
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
        offer_repair = False
        tried_offers = set()
        copy_position_review = None
        layout_attempts = offer_attempts = 0
        measured_fallback = False
        measured_choices = iter(placements)
        used_placements = set()

        def render_copy(design):
            candidate = design_layers(design, copy, size, fonts, stage)
            if photo_cue:
                candidate["action_regions"] = regions
            headline_behind_subject = check_text_overlap(foreground, candidate, stage)
            tighten_text_groups(foreground, candidate, stage)
            ink = rendered_text_bounds(candidate, stage)
            if not all(role in ink for role in TEXT_ROLES):
                raise ValueError("제목과 상품명, 가격이 모두 화면에 보여야 합니다")
            check_offer_spacing(ink, size)
            return candidate, ink, headline_behind_subject

        def render_placement(answer):
            proposed = selected_placement(answer, placements, design)
            styles = format_styles(ai, proposed.get("styles"), layout["texts"], first_lines)
            styles = format_type_hierarchy(styles, design["focus"])
            plan = design | {"artwork_box": [0, 0, 100, 100], "boxes": proposed.get("boxes"),
                             "styles": styles, "accent": proposed.get("accent")}
            plan.pop("subject_box", None)
            rendered, actual_ink, behind = render_copy(plan)
            try:
                check_type_margin(size, actual_ink)
            except ValueError:
                shifted = move_copy_inside(plan, actual_ink, size) if intent else None
                if shifted is None:
                    raise
                rendered, actual_ink, behind = render_copy(shifted)
                check_type_margin(size, actual_ink)
                plan = shifted
            return plan, rendered, actual_ink, behind

        for attempt in range(4):
            joint_repair = False
            if offer_repair:
                options = (offer_position_options(revised, candidate, foreground, stage, tried_offers)
                           if offer_attempts < 2 else {})
                offer_repair = False
                if options:
                    offer_attempts += 1
                    previews = [offer_preview(compose(option[1], stage), option[2]) for option in options.values()]
                    choice = ask_stage(ai, "studio_format_offer", OFFER_POSITION,
                                       {"position_order": list(options), "observation": reading, "copy": copy},
                                       [to_float(offer_preview(poster, ink)), *map(to_float, previews)])
                    position = choice.get("position")
                    if isinstance(position, str) and position in options:
                        revised, candidate, ink, key = options[position]
                        tried_offers.add(key)
                        offer_repair = True
                if not offer_repair:
                    joint_repair = True
                    inputs["layout_error"] = (f"{reading} 상품명과 가격만 옮겨도 문제가 남습니다. "
                                              "사진과 확정 문구는 유지하고 제목까지 함께 재배치하세요.")
            if not offer_repair:
                if layout_attempts == 2:
                    if not measured_fallback:
                        break
                    for name in measured_choices:
                        if name in used_placements:
                            continue
                        answer = {"placement": name}
                        try:
                            revised, candidate, ink, headline_behind_subject = render_placement(answer)
                        except ValueError:
                            continue
                        break
                    else:
                        break
                else:
                    layout_attempts += 1
                    answer = ask_stage(ai, "studio_reframe", REFRAME,
                                       inputs | {"preferred_first_lines": first_lines}, images)
                    if placements and not intent and "layout_error" not in inputs and answer.get("placement") is None:
                        answer = {"placement": next(iter(placements))}
                    if isinstance(name := answer.get("placement"), str):
                        used_placements.add(name)
                    try:
                        revised, candidate, ink, headline_behind_subject = render_placement(answer)
                    except ValueError as error:
                        inputs["layout_error"] = str(error)
                        inputs["previous_layout"] = answer
                        measured_fallback = layout_attempts == 2
                        continue
                framing.save(stage / candidate["artwork"]["file"])
                framing_mask.save(stage / "foreground.png")
                copy_position_review = None
                if joint_repair or (not attempt and (photo_cue or intent)):
                    options = copy_position_options(revised, candidate, ink, foreground, stage)
                    if options:
                        (revised, candidate, ink), copy_position_review = compare_copy_positions(
                            ai, options, (revised, candidate, ink), foreground, stage,
                            {"photo_cue": photo_cue, "intent": intent, "copy": copy})
                tried_offers = {tuple(revised["boxes"][role][axis]
                                      for role in ("product_name", "price") for axis in (0, 1))}
            backdrop = compose(candidate | {"texts": []}, stage) if candidate.get("graphic") else framing
            revised, candidate = readable_headline_colors(revised, candidate, backdrop, stage, foreground)
            revised, candidate = readable_offer_colors(revised, candidate, backdrop, stage)
            poster = compose(candidate, stage)
            phone = ImageOps.contain(poster, (360, 640), Image.Resampling.LANCZOS)
            review = ask_stage(ai, "studio_format_reading", FORMAT_READING,
                               {"copy": copy, "focus": design["focus"], "product_bounds": bounds, "rendered_ink": ink,
                                "outer_margins": outer_margins(bounds, ink.values()),
                                "headline_behind_subject": headline_behind_subject,
                                **({"photo_cue": layout["photo_cue"]} if layout.get("photo_cue") else {}),
                                **({"context_feature": context_feature} if context_feature else {}),
                                **({"intent": intent} if intent else {})},
                               [reference, to_float(phone), to_float(poster), to_float(offer_detail(poster, ink)),
                                to_float(text_detail(poster, (ink["headline"],))),
                                to_float(text_detail(framing, (ink["headline"],)))])
            reading = required_text(review, "observation", "게시 규격 검토")
            finding = review.get("finding")
            touching = review.get("headline_touches_product")
            if finding not in ("ready", "offer", "layout", "photo") or type(touching) is not bool:
                raise ValueError("게시 규격의 글자와 상품 배치 판정을 확인하지 못했습니다")
            if touching and not headline_behind_subject and finding in ("ready", "offer"):
                finding = "layout"
                reading = f"제목이 사진 속 상품 윤곽에 붙어 보입니다. 제목을 윤곽에서 띄워 주세요. {reading}"
            if finding == "photo":
                raise FormatPhotoIssue(reading, poster)
            if finding != "ready":
                inputs.update(layout_error=reading, previous_layout=revised)
                inputs["previous_placement"] = None if offer_repair else answer.get("placement")
                offer_repair = finding == "offer" and revised.get("accent") is None
                images[3:] = [to_float(poster)]
                continue
            poster.save(stage / "ad.png")
            candidate.update(fields=layout["fields"], brief=layout["brief"], design=revised,
                             scene_prompt=layout["scene_prompt"], format=format_name, final_review_reason=reading)
            if copy_position_review:
                candidate["copy_position_review"] = copy_position_review
            elif not offer_repair and answer.get("placement") is not None:
                candidate["text_placement"] = answer["placement"]
            if "art_direction_plan" in layout:
                candidate["art_direction_plan"] = layout["art_direction_plan"]
            if "photo_cue" in layout:
                candidate["photo_cue"] = layout["photo_cue"]
            if context_feature:
                candidate["context_feature"] = context_feature
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
                                "has_attempted_resize": bool(before), "has_staged_group": bool(details),
                                **({"photo_cue": layout["photo_cue"]} if layout.get("photo_cue") else {})},
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
