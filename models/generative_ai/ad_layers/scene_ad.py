"""포스터의 구도를 먼저 정하고 그림과 글자를 별도 레이어로 만듭니다."""

import argparse
import json
import re
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .art_direction import ANGLES, LIGHTING, SHOT_SCALES, THEMES, plan_art_direction
from .creative_brief import LAYOUT_ROLES, TEXT_ROLES, ask_stage, plan_copy, required_text, validate_boxes
from .layers import compose, cutout_product, load_layout, load_settings, save_layout, to_float
from .staging import on_gray, source_parts, stage_group, stage_subject
from .text_render import minimum_font_size, render_texts
from .viewpoint import available_viewpoints, camera_view, place_camera_view


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
"box", "color" #RRGGBB, and "layer" background, overlay or behind_subject), "artwork_box", "subject_box", "edge" (blend or framed),
"crop" (preserve or intentional), and "art_direction" (English
photographic framing, lighting, materials and background for the artwork region; no text or graphic overlays).
Use a background accent outside the artwork, or an overlay accent over quiet space without covering the product.
Use behind_subject when a shape should tuck behind the real silhouette. Give it a clear visible role linking the
feature and headline. The compositor hides only the part behind the subject and keeps the product visible.
Use the blended edge for a calm photo perimeter that can merge into a plain graphic field. A framed edge can
align a deliberate, distinct image panel with the type or cover the whole canvas. When the background above
or below the photo should read as the same continuous surface, prefer blend.
Use preserve for a complete product by default. Choose intentional only when a motivated crop emphasizes a real
detail and the shopper can still identify the product and any important label.
For an intentional detail shot include source_crop as [left, top, width, height] in whole percentages of the
tight product bounds in the original image. Select the specific detail the headline is about. Space around the
product in the original photo is outside these bounds. Use null or omit source_crop for a complete product.
Place intentional cut edges against a photo edge rather than leaving a sliced product floating inside open space.
If there is a quiet area for type over the artwork, say where it is and describe its light and texture in art_direction.
Include "boxes" and "styles", both keyed by headline, product_name and price.
All boxes are [left, top, width, height]. Use whole percentages of the poster from its top-left, except subject_box,
which is relative to the photograph itself. Put the product inside subject_box with the intended scale and space
around the edges. In the shared photo area, reserve space for the headline and the other copy outside the product.
Keep text boxes
inside the canvas, off one another, and large enough for the supplied copy. Text may overlap the artwork box
only where you plan quiet background. Group related text by alignment and use sufficient contrast.
For each style return "font" (one of the supplied fonts), "align" (left, center, right), "color" (#RRGGBB),
"size" (intended font size as a percentage of canvas width), and "max_lines" (1 or 2 for the
headline or product_name, 1 for price). If two lines are intentional, set "break_after" to the number of
space-separated words on the first line. Keep Korean phrases, quantities and product names intact at the
break. For the headline only, optionally set "depth" to "behind_subject" when the contour of the foreground
can deliberately cross a small outer part of a large headline. Every word must stay readable. Leave out
depth when the type belongs in clear space. For the headline only, optional emphasis is an object with phrase
(an exact contiguous phrase from the headline) and color (#RRGGBB). Use one accent when a visible color from
the photograph draws attention to the corresponding words. Keep sufficient contrast. Use null to omit it.
For a deliberately two-line headline, optional styles.headline.lines contains two objects in reading order.
Each has a non-overlapping box inside the headline's outer box, in whole poster percentages, and may override
size, align, color, font or depth. For an override, depth can be only front or behind_subject. Use break_after
to choose the exact words on each line. Set distinct positions
or scales only when each clause has a visible role, such as a quiet lead-in and a short line near the subject.
If a prior design failed, address the reported issue."""

BLOCKING = """Review the first image as a full poster made from the real product silhouette and the exact
rendered typography. The flat background is a placeholder; judge placement, scale, spacing and how the
reader's eye moves. The second image is the source photograph. Any further images are composition references.
Name the specific element that physically connects the headline and subject. If reading the separate blocks
depends on a lighting effect that has not been generated, correct the underlying geometry now. Consider a
common center axis, a relationship to a particular part of the product, or a deliberate edge relationship
with a legible large headline. Use the treatment justified by this product and message.
When the structure is weak, move or resize the actual subject_box and the copy together. Avoid only changing
font size or color. Keep the full recognizable product, exact copy, and one clear group for supporting details.
Read the thumbnail from the first focal point to the offer. Check whether any solid accent makes supporting
information the first focal point. Let that happen when the offer is the approved message. Otherwise soften
or remove the accent and use alignment, spacing or weight to keep the offer together. Use rendered_text_bounds
to check the actual ink edges before the photograph is generated.
If present, follow the requested visual correction. Return JSON with reason (Korean) and revision (null when
the structure works, or an object containing the design fields that need to change)."""

BLOCKING_COMPARE = """Compare the two rendered drafts in the first two images. They use the same product
and copy and neither has final lighting. Any later images are composition references. Judge the visible
relationship between the headline and product, the way their sizes balance, and the placement of the
smaller information. Look at the actual space and ink, even if the proposal described a different effect.
Compare which element gets the first glance and whether any graphic distracts from the advertised detail.
Keep the original when the revision would merely trade one generic arrangement for another.
Return JSON with choice (original or revision) and a concrete Korean reason."""

SOURCE_FRAMING = """Propose one close framing to compare with the whole available product before layout.
The first image is the isolated product in its tight bounds. The second is the original photo; any later images
are references. Another reviewer will choose between the two views after seeing the rendered crop. If the message
concerns one detail or a few neighboring features, show those features together much larger. Leave out parts that
do not help explain the message. A whole shot can be good and still deserve comparison with a close shot.
Keep context to recognize the item and its use. Return null if a meaningful crop would break necessary labels,
shape, count, or an arrangement across the whole product, or if the request specifies keeping the full view.
The original photo may already cut an edge of the object; preserve the meaning of its available view.
Return JSON with source_crop (null or [left, top, width, height] in integer percentages of the first image)
and reason (in Korean, what the crop makes easier to notice, or why a crop cannot help)."""

SOURCE_FRAMING_CHECK = """The first two images show the whole available product and the actual crop in equal-size frames.
The third image is the original photo. Compare the visible pixels against the exact headline and the intended
selling detail. Check recognition, number of items, important labels, and the edges and detail that actually
remain. Favor the crop only when it makes the detail meaningfully more prominent and keeps recognizable context.
Keep the whole view if the crop mainly removes an edge with little practical enlargement of the selling detail.
Return JSON with choice (whole or detail) and a Korean reason grounded in those two images."""

FINAL_REVIEW = """Review the first poster, using the second image (its generated photo without type) and the reference
posters that follow. Keep the generated photo, position, crop and canvas color. Use the actual product bounds to
place type outside the product. Group the related text, give the main line a distinct scale, and judge contrast
using the actual photograph. Find the visible axis, gesture or light that connects photo and words. If there is
an unused band between them or the offer reads as unrelated fragments, move and scale the type as one composition.
Use rendered_text_bounds to judge the actual ink, not the empty text boxes. For stacked supporting copy,
leave less space between the name and price than between their group and the headline. Keep that group near
the product rather than letting all the type drift into a distant corner. For a one-line information group,
fit the first text box to its visible lettering before placing the next box. A shared baseline with excess
space between the two can still make them look unrelated.
Keep useful relationships that already exist. When the original title leads into the photo from above and
supporting copy closes the reading path below it, preserve that order unless the new placement has a stronger
visible reason. Examine the whole canvas before moving every text group into the same margin.
When the approved image and headline still feel like separate columns, consider whether one larger headline
can share the subject's axis or tuck behind a small part of its outline. Use depth only when it adds that relation.
Match the lettering to the visual mood and keep any confirmed line breaks. Use the supplied wording exactly.
When the photograph clearly supports one specific phrase, consider highlighting it with a color sampled visually
from the relevant feature. Put {phrase, color} only at revision.styles.headline.emphasis. The revision.accent
field is only for a graphic shape using shape, layer, box and color, or null to remove the graphic.
For a deliberate shape that links the real subject contour to the headline, use layer behind_subject. The
subject stays entirely visible. Keep both a visible part of the graphic and a part tucked behind the subject.
Copy whole space-separated words exactly, including attached Korean particles or punctuation. Omit the emphasis
if there is no suitable phrase. Use a single treatment and keep the accent legible against the actual background.
If the same bright color on supporting information competes with the headline, reduce its emphasis.
When a single headline box keeps both clauses far from the advertised feature, use an intentional break_after
and two styles.headline.lines objects. Select an integer break_after from headline_breaks, which lists the first
and second line text for each valid break. Each line has its own box in whole poster percentages inside the outer
headline box, and can override size, font, align, color or depth. Place the line that names the feature near it.
Use the other line to set up the message. Keep the two line boxes in reading order, vertical and non-overlapping.
Keep both lines readable at feed size. Leave lines null to use one shared headline style.
If a solid graphic panel is the strongest element by color contrast and the offer is secondary, soften or
remove the panel. Use a common alignment and gap for the supporting text instead of relying on a shared box.
If a hard photo boundary behind the words or between the heading and product cuts a shared background for no
clear reason, set edge to blend so only the outer surroundings fade into the canvas. Inspect the top and bottom
edges as well as the sides. Keep framed when a distinct image panel improves the composition. This affects the
surroundings, leaving the subject intact.
Return one JSON object with exactly two keys: reason (a concrete Korean sentence) and revision. A revision
contains only boxes, styles, accent or edge in the same form as the supplied design. Each of boxes and styles,
if changed, must include headline, product_name and price. Boxes are [left, top, width, height] in whole
percentages. Their rectangles must not intersect. The end of one stacked box must be at or before the next
box's top. Use depth only for the value behind_subject and omit it to put type in front. Fix any reported
layout_error using the actual photograph. Return revision: null only when the layout works and layout_error is null."""

FINISHED_COMPOSITION_COMPARE = """The first poster is the current one, and the second tries to fix the stated
composition_issue. Compare the actual letter shapes, gaps, and their relationship to the photographed feature.
Use the visible ink bounds to locate the type. Count a change in position, size or color only when it makes
the reading path or hierarchy clearer.
When balanced_outer_space is present, image three keeps the first revision's type treatment with more modest
placement changes. Compare all three actual posters before choosing. Any reference images come afterward.
Start by looking at the full canvas in each poster. Use original_outer_space and revision_outer_space to inspect
the areas outside both the words and product, especially the ends of the long axis. Moving text toward the
product can reduce internal gaps while creating two bigger empty bands at the edges. Keep the current
poster if the revision creates a large purposeless band, even when neighboring elements are closer.
An enlarged margin can work if it visibly supports the photographed action, guides the eye or gives an
important element emphasis. Explain that visible purpose when choosing a revision with more outer space.
Use product_bounds alongside both ink maps. When the headline moves from above the product to below it or
the reverse, explain what that move does to the entire reading path and the margin it leaves behind.
Choose revision only when the affected parts now read together more clearly or the issue is visibly solved.
If the request identifies several related problems, check each in both images. Keep the current poster if a
requested relationship still looks broken or if the change solves one by worsening another.
If the copy is readable in both, explain what happens to the advertised feature and title on the first glance.
Return choice (original, revision, or balanced when image three is an alternative) and a concrete Korean reason
that describes what the images show."""

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
planned type. Translate the selected lighting into visible highlights and shadows on named parts of the product.
Keep the product prominent and recognizable, with its visible details, packaging and labels.
Describe the actual foreground item and its visible contents concretely. Preserve the visible kinds of items.
If the product name is broader than the photograph, do not fill in other items, varieties or components.
When there is a product demonstration, describe exactly what supports each object and the physical relationship
of the moving parts. Use surroundings and props only where they serve this particular image or the owner's
request. Follow the agreed crop. State only what the generated photograph should show. Include no ad text,
graphics, coordinates or layout markup. Return JSON with one string, 'scene'."""

FORMAT_STAGING = """The first image is the source product, the second the approved campaign, and the
third the isolated original product. Choose whether to physically move one already visible, independently
movable piece in the source to create a better reading path for target_canvas. Look at how the current
group runs across the source. If it sits on the wrong axis, consider pulling one real piece forward or
sliding it sideways on the same surface to give the group depth or direction. Preserve all product
variants, count, proportions and the campaign's visible motif. Do not propose moving an attached part
or changing size, camera angle, or the surface. Return {"move": "name the actual piece, new location and
desired visible geometry in one staging instruction"} or {"move": null} if the existing arrangement fits
or a safe move cannot resolve format_issue."""


def is_color(value):
    return isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value) is not None


def valid_headline_emphasis(emphasis, text):
    phrase = emphasis.get("phrase") if isinstance(emphasis, dict) else None
    return (isinstance(phrase, str) and bool(phrase) and phrase.strip() == phrase and phrase != text
            and text.count(phrase) == 1 and re.search(r"(?<!\S)" + re.escape(phrase) + r"(?!\S)", text) is not None
            and is_color(emphasis.get("color")))


def pixel_box(box, size):
    left, top, width, height = box
    canvas_w, canvas_h = size
    x, y = round(left * canvas_w / 100), round(top * canvas_h / 100)
    return (x, y, round((left + width) * canvas_w / 100) - x,
            round((top + height) * canvas_h / 100) - y)


def text_style(style, fonts, role):
    font, align, color, size = (style.get(key) for key in ("font", "align", "color", "size"))
    if (font not in tuple(fonts) or align not in ("left", "center", "right") or not is_color(color)
            or type(size) not in (int, float) or not 2 <= size <= 20):
        raise ValueError(f"{role} 글자 스타일 값이 올바르지 않습니다")
    return {"font": font, "align": align, "color": color}


def headline_breaks(text):
    words = text.split()
    return [{"break_after": index, "first_line": " ".join(words[:index]), "second_line": " ".join(words[index:])}
            for index in range(1, len(words))]


def headline_lines(layer, parent_style, canvas_size, fonts):
    styles = parent_style["lines"]
    if (not isinstance(styles, list) or len(styles) != 2 or "break_after" not in layer
            or any(not isinstance(style, dict)
                   or style.keys() - {"box", "size", "font", "align", "color", "depth", "text"}
                   for style in styles)):
        raise ValueError("나눠 배치할 제목에는 줄바꿈과 두 줄의 위치 및 스타일을 지정해야 합니다")
    boxes = {str(index): style.get("box") for index, style in enumerate(styles, 1)}
    validate_boxes(boxes, tuple(boxes))
    if boxes["1"][1] + boxes["1"][3] > boxes["2"][1]:
        raise ValueError("제목의 첫 줄은 둘째 줄보다 위에 있어야 합니다")
    text = layer["text"]
    words = list(re.finditer(r"\S+", text))
    split = layer["break_after"]
    parts = [(0, words[split - 1].end()), (words[split].start(), len(text))]
    output = []
    for index, ((start, end), style) in enumerate(zip(parts, styles, strict=True), 1):
        line_text = text[start:end]
        if style.get("text", line_text) != line_text:
            raise ValueError("제목 줄의 문구가 확정된 원문과 다릅니다")
        x, y, w, h = pixel_box(boxes[str(index)], canvas_size)
        if not (layer["x"] <= x < x + w <= layer["x"] + layer["w"]
                and layer["y"] <= y < y + h <= layer["y"] + layer["h"]):
            raise ValueError("제목 각 줄은 제목 전체 영역 안에 들어가야 합니다")
        visual = parent_style | style
        child = text_style(visual, fonts, "제목 줄") | {
            "text": line_text, "role": f"headline_line_{index}", "x": x, "y": y, "w": w, "h": h,
            "file": f"text_headline_{index}.png", "max_lines": 1,
            "max_font_size": round(visual["size"] * canvas_size[0] / 100), "min_font_size": layer["min_font_size"],
        }
        depth = style.get("depth", layer.get("depth"))
        if depth not in (None, "front", "behind_subject"):
            raise ValueError("제목 줄의 깊이는 제품 앞 또는 뒤여야 합니다")
        if depth == "behind_subject":
            child["depth"] = depth
        if emphasis := layer.get("emphasis"):
            marked = text.index(emphasis["phrase"])
            first, last = max(start, marked), min(end, marked + len(emphasis["phrase"]))
            if first < last:
                child["emphasis"] = {"phrase": text[first:last], "color": emphasis["color"], "start": first - start}
        output.append(child)
    return output


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
        visual = text_style(style, fonts, role)
        size, max_lines, break_after = (style.get(key) for key in ("size", "max_lines", "break_after"))
        if type(max_lines) is not int or not 1 <= max_lines <= (1 if role == "price" else 2):
            raise ValueError(f"{role} 글자 스타일 값이 올바르지 않습니다")
        layer = visual | {"role": role, "text": text,
                 "x": x, "y": y, "w": w, "h": h, "file": f"text_{role}.png", "max_lines": max_lines,
                 "max_font_size": round(size * width / 100), "min_font_size": minimum_font_size(width)}
        depth = style.get("depth")
        if depth not in (None, "front"):
            if role != "headline" or depth != "behind_subject":
                raise ValueError("제품 뒤에 걸 수 있는 글자는 제목뿐입니다")
            layer["depth"] = depth
        if break_after is not None:
            if max_lines != 2 or type(break_after) is not int or not 0 < break_after < len(text.split()):
                raise ValueError(f"{role} 줄바꿈 위치가 올바르지 않습니다")
            layer["break_after"] = break_after
        emphasis = style.get("emphasis")
        if emphasis is not None:
            if role != "headline" or not valid_headline_emphasis(emphasis, text):
                raise ValueError("강조는 제목에 한 번 나오는 일부 어절과 올바른 색을 지정해야 합니다")
            layer["emphasis"] = {key: emphasis[key] for key in ("phrase", "color")}
        if style.get("lines") is not None:
            if role != "headline":
                raise ValueError("줄마다 따로 배치할 수 있는 문구는 제목입니다")
            layer["lines"] = headline_lines(layer, style, canvas_size, fonts)
            del layer["file"]
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
    if answer.get("source_crop") is not None:
        if answer["crop"] != "intentional":
            raise ValueError("원본 일부를 확대하려면 의도한 잘림으로 지정하세요")
        validate_boxes({"source_crop": answer["source_crop"]}, ("source_crop",))
    x, y, w, h = pixel_box(answer["artwork_box"], size)
    if min(answer["artwork_box"][2:]) < 35:
        raise ValueError("상품 사진을 보여 줄 영역이 너무 작습니다")
    background = Image.new("RGB", size, answer["background"])
    graphic = None
    if accent is not None:
        if (not isinstance(accent, dict) or accent.get("shape") not in ("rectangle", "ellipse")
                or accent.get("layer", "background") not in ("background", "overlay", "behind_subject")
                or not is_color(accent.get("color"))):
            raise ValueError("색면의 형태와 색이 올바르지 않습니다")
        validate_boxes({"accent": accent.get("box")}, ("accent",))
        ax, ay, aw, ah = pixel_box(accent["box"], size)
        overlay = accent.get("layer") in ("overlay", "behind_subject")
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
            if accent["layer"] == "behind_subject":
                graphic["depth"] = "behind_subject"
            target.save(folder / graphic["file"])
    texts = layout_texts(answer, copy, size, fonts)
    render_texts(texts, size, folder)
    background.save(folder / "background.png")
    layout = {"size": list(size), "background": {"file": "background.png"}, "texts": texts,
              "artwork": {"file": "scene.png", "x": x, "y": y, "w": w, "h": h}}
    if graphic:
        layout["graphic"] = graphic
    return layout


def extract_subject(photo, mask_model, threshold):
    from ..photo_retouch.product_mask import product_mask

    mask = product_mask(mask_model, to_float(photo))
    subject = Image.fromarray(mask > threshold)
    product = cutout_product(photo, mask, threshold) if subject.getbbox() else None
    return product, subject


def crop_subject(product, design):
    if product is None or design.get("source_crop") is None:
        return product
    x, y, width, height = pixel_box(design["source_crop"], product.size)
    detail = product.crop((x, y, x + width, y + height))
    if not detail.getchannel("A").getbbox():
        raise ValueError("확대할 부분에 원본 상품이 없습니다")
    return detail


def choose_source_framing(ai, product, inputs, images):
    whole = on_gray(product)
    answer = ask_stage(ai, "ad_source_framing", SOURCE_FRAMING, inputs, [to_float(whole), *images])
    crop = answer.get("source_crop")
    reason = answer.get("reason")
    if "source_crop" not in answer or not isinstance(reason, str) or not reason.strip():
        return {"source_crop": None, "reason": "추천 구도를 확인하지 못해 원본 범위를 유지합니다"}
    if crop is None or crop == [0, 0, 100, 100]:
        return {"source_crop": None, "reason": reason}
    try:
        validate_boxes({"source_crop": crop}, ("source_crop",))
        detail = crop_subject(product, answer)
    except ValueError as error:
        return {"source_crop": None, "reason": f"{error}. 원본 범위를 유지합니다"}
    frames = [to_float(ImageOps.pad(view, (512, 512), Image.Resampling.LANCZOS, color="#808080"))
              for view in (whole, on_gray(detail))]
    review = ask_stage(ai, "ad_source_framing_check", SOURCE_FRAMING_CHECK, inputs | {"proposal": reason},
                       [*frames, images[0]])
    review_reason = review.get("reason")
    if review.get("choice") not in ("whole", "detail") or not isinstance(review_reason, str) or not review_reason.strip():
        return {"source_crop": None, "reason": "확대본의 비교 결과가 없어 원본 범위를 유지합니다"}
    return {"source_crop": crop if review["choice"] == "detail" else None, "reason": review_reason}


def frame_subject(product, picture_size, subject_box, source_crop=None):
    x, y, width, height = pixel_box(subject_box, picture_size)
    contained = ImageOps.contain(product, (width, height), Image.Resampling.LANCZOS)
    extra_width, extra_height = width - contained.width, height - contained.height
    offset_x, offset_y = extra_width // 2, extra_height // 2
    if source_crop:
        left, top, crop_width, crop_height = source_crop
        if left > 0 and left + crop_width == 100:
            offset_x = 0
        elif left == 0 and crop_width < 100:
            offset_x = extra_width
        if top > 0 and top + crop_height == 100:
            offset_y = 0
        elif top == 0 and crop_height < 100:
            offset_y = extra_height
    frame = Image.new("RGBA", picture_size)
    frame.alpha_composite(contained, (x + offset_x, y + offset_y))
    return frame


def product_reference(photo, product, picture_size, subject_box, source_crop=None):
    if product is None:
        return photo
    if subject_box:
        reference = Image.new("RGB", picture_size, "#808080")
        framed = frame_subject(product, picture_size, subject_box, source_crop)
        reference.paste(framed, mask=framed.getchannel("A"))
        return reference
    margin = round(max(product.size) / 10)
    reference = Image.new("RGB", (product.width + 2 * margin, product.height + 2 * margin), "#808080")
    reference.paste(product, (margin, margin), product)
    return reference


def composition_preview(product, design, layout, folder):
    artwork = layout["artwork"]
    size = (artwork["w"], artwork["h"])
    subject = frame_subject(crop_subject(product, design), size, design.get("subject_box", [8, 8, 84, 84]),
                            design.get("source_crop"))
    image = Image.new("RGBA", size, design["background"])
    image.alpha_composite(subject)
    image.save(folder / artwork["file"])
    foreground = np.asarray(subject.getchannel("A")) > 128
    Image.fromarray(foreground).save(folder / "foreground.png")
    return compose(layout, folder), foreground


def touches_photo_edge(mask):
    return mask[[0, -1], :].any() or mask[:, [0, -1]].any()


def measured_format_composition(ai, design, layout, folder, foreground, first, inputs):
    from .format_layout import format_measurements, measured_styles, position_options, preview_placement

    size = tuple(layout["size"])
    artwork = layout["artwork"]
    x, y, w, h = (artwork[key] for key in ("x", "y", "w", "h"))
    mask = np.zeros((size[1], size[0]), bool)
    mask[y:y + h, x:x + w] = foreground
    box = Image.fromarray(mask).getbbox()
    if box is None:
        return None
    bounds = [round(100 * edge / limit) for edge, limit in zip(box, size * 2, strict=True)]
    measured = layout | {"design": design}
    options = position_options(measured, size, mask, bounds, format_measurements(measured, size))
    if not options:
        return None
    artwork_only = compose(layout | {"texts": [], "graphic": None}, folder)
    previews = [to_float(preview_placement(artwork_only, layout, folder, plan)) for plan in options.values()]
    answer = ask_stage(ai, "ad_format_measured_layout",
                       "Image 1 is the rejected native layout. The remaining images use the same actual subject "
                       "and exact legible words, placed near its silhouette in option_order. Choose the one that "
                       "visibly resolves format_issue and layout_error. Consider both the lines of the headline "
                       "and the product name and price as groups. Check the full frame on the long axis. "
                       "Return placement as one option key or null if none works, and reason in Korean.",
                       {"option_order": list(options), "format_issue": inputs["revision_feedback"]["format_issue"],
                        "layout_error": inputs.get("layout_error"), "copy": inputs["copy"]},
                       [to_float(first), *previews])
    placement = answer.get("placement")
    if not isinstance(placement, str) or placement not in options:
        return None
    plan = options[placement]
    revised = design | {"boxes": plan["boxes"], "styles": measured_styles(design, plan), "accent": None}
    return revised, required_text(answer, "reason", "게시 규격의 실제 글자 배치")


def review_composition_plan(ai, product, design, layout, folder, inputs, images):
    first, foreground = composition_preview(product, design, layout, folder)
    photo_input = [to_float(first), *images]
    review_input = inputs | {"design": design, "rendered_text_bounds": rendered_text_bounds(layout, folder)}
    task = BLOCKING + "\n" + DESIGN_SCHEMA
    format_issue = inputs.get("revision_feedback", {}).get("format_issue")
    if format_issue:
        task += ("\nThe draft is for the target canvas_size. Check the visible geometry against format_issue. "
                 "If the same gap or disconnected offer remains, revise the shared geometry, including the "
                 "first focal point and negative space. Future lighting alone is not a geometry change. "
                 "When crop is preserve, keep a clear edge around the actual silhouette. Increase its span "
                 "along the long axis using composition, not stretching or squeezing every edge. "
                 "When returning revision null, also return ready true only if the actual draft resolves the issue.")
    if "layout_device" in inputs.get("revision_feedback", {}):
        task += ("\nTest how revision_feedback.layout_device appears in the rendered draft. "
                 "If it only exists in the written plan, move or resize the artwork, lettering or graphic "
                 "until the visible relationship is present. Judge the intended effect at a small size.")
    if "source_framing" in inputs:
        task += "\nKeep the chosen source_framing. Adjust placement, scale and type around that photographed view."

    def check_draft(mask, plan, candidate_layout, candidate_folder):
        check_text_overlap(mask, candidate_layout, candidate_folder)
        if format_issue and plan["crop"] == "preserve" and touches_photo_edge(mask):
            raise ValueError("전체를 보여 줄 상품이 사진 가장자리에 닿습니다")

    editable = {"concept", "focus", "background", "accent", "artwork_box", "subject_box", "crop", "source_crop",
                "edge", "boxes", "styles"}
    if "source_framing" in inputs:
        editable -= {"crop", "source_crop"}
    original_valid = True
    try:
        check_draft(foreground, design, layout, folder)
    except ValueError as error:
        original_valid = False
        review_input["layout_error"] = str(error)

    kept_reason = None
    for _ in range(2):
        review = ask_stage(ai, "ad_composition", task, review_input, photo_input)
        revision = review.get("revision")
        if not isinstance(revision, dict):
            if format_issue:
                observation = required_text(review, "reason", "게시 규격 촬영 전 검토")
                if review.get("ready") is not True or not original_valid:
                    review_input["layout_error"] = observation
                    continue
                kept_reason = observation
            break
        changed = design | {key: revision[key] for key in editable & revision.keys()}
        if changed == design:
            if format_issue:
                review_input["layout_error"] = "현재 배열에서 달라진 항목이 없습니다"
                continue
            break
        with TemporaryDirectory(dir=folder) as temporary:
            candidate = Path(temporary)
            try:
                revised = design_layers(changed, inputs["copy"], tuple(layout["size"]), inputs["fonts"], candidate)
                rendered, proposed_foreground = composition_preview(product, changed, revised, candidate)
                check_draft(proposed_foreground, changed, revised, candidate)
            except ValueError as error:
                review_input.update(rejected_design=changed, layout_error=str(error))
                continue
            reason = required_text(review, "reason", "촬영 전 구도 검토")
            if original_valid:
                compare_task = BLOCKING_COMPARE
                compare_inputs = {"proposed_change": reason, "copy": inputs["copy"]}
                if format_issue:
                    compare_task += (" Return choice neither if both drafts repeat format_issue or only change "
                                     "position without visibly solving the reading path.")
                    compare_inputs["format_issue"] = format_issue
                comparison = ask_stage(ai, "ad_composition_compare", compare_task, compare_inputs,
                                       [to_float(first), to_float(rendered), *images[1:]])
                choice = comparison.get("choice")
                if choice == "neither" and format_issue:
                    review_input["layout_error"] = required_text(comparison, "reason", "게시 규격 구도 비교")
                    review_input["rejected_design"] = changed
                    photo_input = [to_float(first), *images, to_float(rendered)]
                    task += "\nThe final image shows a rejected placement. Address layout_error without repeating it."
                    continue
                if choice == "original":
                    kept_reason = required_text(comparison, "reason", "촬영 전 구도 비교")
                    break
                if choice != "revision":
                    raise ValueError("촬영할 구도 시안을 고르지 못했습니다")
                reason = required_text(comparison, "reason", "촬영 전 구도 비교")
            first.save(folder / "before-composition.png")
            rendered.save(folder / "composition.png")
            for path in candidate.iterdir():
                path.replace(folder / path.name)
            if "graphic" in layout and "graphic" not in revised:
                (folder / layout["graphic"]["file"]).unlink()
            return changed, revised, reason
    if format_issue and not kept_reason:
        anchored = None
        if design["crop"] != "preserve" or not touches_photo_edge(foreground):
            anchored = measured_format_composition(ai, design, layout, folder, foreground, first, review_input)
        if not anchored:
            raise ValueError(f"게시 규격에 맞는 촬영 전 구도를 찾지 못했습니다. {review_input.get('layout_error', format_issue)}")
        changed, kept_reason = anchored
        anchored_layout = design_layers(changed, inputs["copy"], tuple(layout["size"]), inputs["fonts"], folder)
        rendered, foreground = composition_preview(product, changed, anchored_layout, folder)
        check_draft(foreground, changed, anchored_layout, folder)
        if graphic := layout.get("graphic"):
            (folder / graphic["file"]).unlink()
        first.save(folder / "before-composition.png")
        rendered.save(folder / "composition.png")
        return changed, anchored_layout, kept_reason
    first.save(folder / "composition.png")
    return design, layout, kept_reason


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
    layers = [(f"{line['role']} 글자", line) for text in layout["texts"] for line in text.get("lines", (text,))]
    if graphic := layout.get("graphic"):
        layers.append(("색면", graphic))
    for name, layer_info in layers:
        with Image.open(folder / layer_info["file"]) as layer:
            ink = np.asarray(layer.getchannel("A")) > 128
        x, y = layer_info["x"], layer_info["y"]
        covered = mask[y:y + layer_info["h"], x:x + layer_info["w"]] & ink
        if layer_info.get("depth") == "behind_subject":
            if layer_info is graphic:
                if not covered.any() or not (ink & ~covered).any():
                    raise ValueError("제품 뒤 색면은 제품과 맞닿고 바깥에서도 보여야 합니다")
                continue
            if covered.any():
                from scipy.ndimage import label

                parts, _ = label(ink, structure=np.ones((3, 3), dtype=bool))
                sizes = np.bincount(parts[ink])
                hidden = np.bincount(parts[covered], minlength=len(sizes))
                if covered.sum() > 0.12 * ink.sum() or np.any((sizes > 2) & (hidden > 0.35 * sizes)):
                    raise ValueError("제품 뒤로 넘어가는 제목이 너무 많아 읽기 어렵습니다")
        elif covered.any():
            raise ValueError(f"{name} 레이어가 실제 제품을 가립니다")


def tighten_aligned_stack(lines, foreground, layout, folder):
    if (len(lines) != 2 or any(line.get("depth") == "behind_subject" for line in lines)
            or lines[0]["y"] + lines[0]["h"] > lines[1]["y"]):
        return
    paths = [folder / line["file"] for line in lines]
    original = []
    for path in paths:
        with Image.open(path) as layer:
            original.append(layer.convert("RGBA"))
    first, second = (layer.getchannel("A").getbbox() for layer in original)
    if first is None or second is None:
        return
    left, right = [(line["x"] + box[0], line["x"] + box[2])
                   for line, box in zip(lines, (first, second), strict=True)]
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap * 2 < min(left[1] - left[0], right[1] - right[0]):
        return
    desired = max(1, round(0.4 * min(first[3] - first[1], second[3] - second[1])))
    gap = lines[1]["y"] + second[1] - lines[0]["y"] - first[3]
    excess = max(0, gap - desired)
    # Keep the lower line as close as possible to its planned position.
    down = min(excess, original[0].height - first[3])
    up = min(excess - down, second[1])
    if not (down or up):
        return
    for path, layer, box, shift in zip(paths, original, (first, second), (down, -up), strict=True):
        shifted = Image.new("RGBA", layer.size)
        shifted.paste(layer.crop(box), (box[0], box[1] + shift))
        shifted.save(path)
    try:
        check_text_overlap(foreground, layout, folder)
    except ValueError:
        for path, layer in zip(paths, original, strict=True):
            layer.save(path)


def tighten_text_groups(foreground, layout, folder, changed=None):
    roles = {text["role"]: text for text in layout["texts"]}
    if changed is None or "headline" in changed:
        tighten_aligned_stack(roles["headline"].get("lines", ()), foreground, layout, folder)
    if changed is None or {"product_name", "price"} <= changed.keys():
        offer = tuple(sorted((roles["product_name"], roles["price"]), key=lambda text: text["y"]))
        tighten_aligned_stack(offer, foreground, layout, folder)


def rendered_text_bounds(layout, folder):
    size = tuple(layout["size"])
    bounds_by_role = {}
    for text in layout["texts"]:
        line_bounds = []
        for line in text.get("lines", (text,)):
            with Image.open(folder / line["file"]) as image:
                bounds = image.getchannel("A").getbbox()
            if bounds:
                ink = [round(100 * (edge + offset) / side)
                       for edge, offset, side in zip(bounds, (line["x"], line["y"]) * 2, size * 2, strict=True)]
                bounds_by_role[line["role"]] = ink
                line_bounds.append(ink)
        if "lines" in text and line_bounds:
            bounds_by_role[text["role"]] = [min(b[0] for b in line_bounds), min(b[1] for b in line_bounds),
                                             max(b[2] for b in line_bounds), max(b[3] for b in line_bounds)]
    return bounds_by_role


def balanced_text_positions(original, revision, size, before, after):
    axis, ends = (1, ("top", "bottom")) if size[1] >= size[0] else (0, ("left", "right"))
    if any(after[edge] <= before[edge] for edge in ends):
        return None
    offsets = {role: round((original["boxes"][role][axis] - box[axis]) / 2)
               for role, box in revision["boxes"].items()}

    def shift(box, amount):
        moved = box.copy()
        moved[axis] += amount
        return moved

    styles = revision["styles"].copy()
    headline = styles["headline"]
    if "lines" in headline:
        styles["headline"] = headline | {
            "lines": [line | {"box": shift(line["box"], offsets["headline"])} for line in headline["lines"]]}
    return revision | {"styles": styles,
                       "boxes": {role: shift(box, offsets[role]) for role, box in revision["boxes"].items()}}


def review_finished(ai, answer, layout, scene, foreground, folder, design_input, references, *,
                    assess_composition=False, keep_wrap=False):
    """완성 사진 위의 글자와 구도를 검사하고 사진을 유지한 채 배치를 고칩니다."""
    original_error = None
    try:
        check_text_overlap(foreground, layout, folder)
        tighten_text_groups(foreground, layout, folder)
    except ValueError as error:
        original_error = str(error)
    if original_error is None and not assess_composition:
        return answer, layout, None
    first = compose(layout, folder)
    artwork = layout["artwork"]
    size = tuple(layout["size"])
    rows, columns = np.nonzero(foreground)
    placement = [round(100 * value / limit) for value, limit in zip(
        (artwork["x"] + columns.min(), artwork["y"] + rows.min(), artwork["x"] + columns.max() + 1,
         artwork["y"] + rows.max() + 1), (*size, *size), strict=True)]
    inputs = design_input | {"design": answer, "product_bounds_left_top_right_bottom": placement,
                             "headline_breaks": headline_breaks(design_input["copy"]["headline"]),
                             "layout_error": original_error, "rendered_text_bounds": rendered_text_bounds(layout, folder)}
    task = FINAL_REVIEW
    needs_alternative = "composition_issue" in inputs
    if needs_alternative:
        from .format_layout import outer_margins

        inputs["original_outer_space"] = outer_margins(placement, inputs["rendered_text_bounds"].values())
        task += ("\nAddress composition_issue on the actual poster. Design the reading path as a whole: place the "
                 "headline by the photographic feature, then align and group the name and price so they do not "
                 "form an unrelated block. Blend an arbitrary seam that divides the reading path. If a panel "
                 "is appropriate, use its edge to align the elements. Give the headline enough scale for the phone feed. "
                 "Check the whole canvas after moving type, including the larger outer margins a move can leave. "
                 "Return an actual alternative with at least one changed field. Both rendered posters will be compared later.")

    def render_revision(plan, folder):
        updated = design_layers(plan, design_input["copy"], size, design_input["fonts"], folder)
        revised_scene = (scene if plan["edge"] == answer["edge"] else
                         set_photo_edge(scene, foreground, artwork, size, plan["edge"]))
        revised_scene.save(folder / artwork["file"])
        Image.fromarray(foreground).save(folder / "foreground.png")
        check_text_overlap(foreground, updated, folder)
        tighten_text_groups(foreground, updated, folder)
        return updated

    editable = ("boxes", "styles", "accent", "edge")
    for _ in range(2):
        review = ask_stage(ai, "ad_final_layout", task, inputs,
                           [to_float(first), to_float(scene), *references])
        revision = review.get("revision")
        if "revision" not in review:
            revision = {key: review[key] for key in editable if key in review} or None
        if not isinstance(revision, dict):
            if original_error is None and not needs_alternative:
                break
            inputs["layout_error"] = "제출한 배치 수정이 빠졌습니다."
            if original_error:
                inputs["layout_error"] = f"{original_error}. {inputs['layout_error']}"
            continue
        accent = revision.get("accent")
        styles = revision.get("styles", answer["styles"])
        if (isinstance(accent, dict) and accent.keys() == {"phrase", "color"}
                and isinstance(styles, dict) and isinstance(styles.get("headline"), dict)):
            revision = {key: value for key, value in revision.items() if key != "accent"}
            revision["styles"] = styles | {"headline": {"emphasis": accent} | styles["headline"]}
        revised = answer | {key: revision[key] for key in editable if key in revision}
        styles = revised["styles"]
        headline = styles.get("headline") if isinstance(styles, dict) else None
        if isinstance(headline, dict) and (emphasis := headline.get("emphasis")) is not None:
            if not valid_headline_emphasis(emphasis, design_input["copy"]["headline"]):
                headline = headline.copy()
                headline.pop("emphasis")
                original = answer["styles"]["headline"]
                if "emphasis" in original:
                    headline["emphasis"] = original["emphasis"]
                revised["styles"] = styles | {"headline": headline}
        if keep_wrap and isinstance(revised["styles"], dict):
            styles = {}
            for role, style in revised["styles"].items():
                original = answer["styles"].get(role)
                if isinstance(style, dict) and original:
                    style = {key: value for key, value in style.items() if key not in ("max_lines", "break_after")}
                    style.update({key: original[key] for key in ("max_lines", "break_after") if key in original})
                    if "emphasis" in original:
                        style.setdefault("emphasis", original["emphasis"])
                styles[role] = style
            revised["styles"] = styles
        if revised == answer:
            if original_error is None and not needs_alternative:
                break
            inputs["layout_error"] = "이전과 같은 배치입니다."
            if original_error:
                inputs["layout_error"] = f"{original_error}. {inputs['layout_error']}"
            continue
        with TemporaryDirectory(dir=folder) as staging:
            candidate = Path(staging) / "revision"
            candidate.mkdir()
            try:
                updated = render_revision(revised, candidate)
            except ValueError as error:
                inputs = inputs | {"rejected_design" if needs_alternative else "design": revised,
                                   "layout_error": str(error)}
                continue
            if needs_alternative and original_error is None:
                revision_ink = rendered_text_bounds(updated, candidate)
                evidence = {
                    "composition_issue": inputs["composition_issue"], "copy": inputs["copy"],
                    "product_bounds": placement,
                    "original_ink": inputs["rendered_text_bounds"],
                    "revision_ink": revision_ink,
                    "original_outer_space": inputs["original_outer_space"],
                    "revision_outer_space": outer_margins(placement, revision_ink.values()),
                }
                versions = {"revision": (revised, updated, candidate)}
                pictures = [to_float(first), to_float(compose(updated, candidate))]
                balanced = balanced_text_positions(answer, revised, size, evidence["original_outer_space"],
                                                   evidence["revision_outer_space"])
                if balanced is not None and balanced != revised:
                    alternate = Path(staging) / "balanced"
                    alternate.mkdir()
                    try:
                        balanced_layout = render_revision(balanced, alternate)
                    except ValueError:
                        pass
                    else:
                        balanced_ink = rendered_text_bounds(balanced_layout, alternate)
                        evidence.update(balanced_ink=balanced_ink,
                                        balanced_outer_space=outer_margins(placement, balanced_ink.values()))
                        versions["balanced"] = (balanced, balanced_layout, alternate)
                        pictures.append(to_float(compose(balanced_layout, alternate)))
                comparison = ask_stage(ai, "ad_final_composition_compare", FINISHED_COMPOSITION_COMPARE, evidence,
                                       [*pictures, *references])
                reason = required_text(comparison, "reason", "완성 배치 비교")
                choice = comparison.get("choice")
                if choice == "original":
                    inputs = inputs | {"rejected_design": revised, "layout_error": reason}
                    continue
                if not isinstance(choice, str) or choice not in versions:
                    raise ValueError("어느 배치가 더 나은지 확인하지 못했습니다")
                revised, updated, candidate = versions[choice]
                review["reason"] = reason
            first.save(folder / "before-typesetting.png")
            for path in candidate.iterdir():
                path.replace(folder / path.name)
            if "graphic" in layout and "graphic" not in updated:
                (folder / layout["graphic"]["file"]).unlink()
            return revised, updated, review.get("reason")
    if original_error is not None:
        raise ValueError(original_error)
    if needs_alternative:
        raise ValueError(inputs["layout_error"])
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


def set_photo_edge(scene, foreground, artwork, size, edge):
    revised = (blend_edges(scene, position=(artwork["x"], artwork["y"]), canvas_size=size)
               if edge == "blend" else scene.convert("RGB").convert("RGBA"))
    alpha = revised.getchannel("A")
    alpha.paste(scene.convert("RGBA").getchannel("A"), mask=Image.fromarray(foreground))
    revised.putalpha(alpha)
    return revised


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
                  rejected=None, feedback=None, scene_issue=None, start_direction=None, viewpoint=None,
                  campaign_intent=None, format_rebuild=None):

    if folder.exists():
        raise FileExistsError(folder)
    settings = load_settings()
    ai = models["ai"]
    if proposal and previous:
        raise ValueError("저장한 시안 적용과 완성본 개선 중 하나를 선택하세요")
    if rejected and not previous:
        raise ValueError("보류한 후보를 전달하려면 비교 기준이 되는 완성본도 필요합니다")
    if feedback and not previous:
        raise ValueError("장면을 수정하려면 기준이 되는 완성본이 필요합니다")
    if campaign_intent and (not previous or feedback):
        raise ValueError("새 연출에는 기존 포스터와 하나의 연출 의도가 필요합니다")
    if format_rebuild is not None and (not previous or feedback or campaign_intent or rejected or viewpoint):
        raise ValueError("게시 규격에 맞춘 새 구도에는 채택한 포스터가 필요합니다")
    prior = load_layout(previous) if previous else None
    kept = prior or proposal
    if kept and kept["fields"] != fields:
        raise ValueError("시안의 상품 정보가 현재 입력과 다릅니다")
    brief = kept["brief"] if kept else plan_copy(ai, photo, fields, idea, start_direction=start_direction)
    copy = {"headline": brief["headline"], "product_name": fields["product_name"],
            "price": f"{fields['price_krw']:,}원"}
    canvas = settings["canvas"]
    size = (canvas["width"], canvas["height"])
    if format_rebuild is not None:
        size, format_issue = format_rebuild
        copy = {text["role"]: text["text"] for text in prior["texts"]}
    images = [to_float(photo), *map(to_float, references)]
    revision_info = {"requested_scene_change": feedback} if feedback else None
    design_images = images
    mask_model = models.get("mask")
    threshold = settings["product"]["mask_threshold"]
    product, source_mask = None, None
    parts = None
    staging_intent = None
    if prior:
        if prior.get("preview"):
            raise ValueError("사진 생성이 끝난 광고만 개선에 사용할 수 있습니다")
        with Image.open(previous / "ad.png") as finished:
            prior_image = to_float(finished)
        if feedback or format_rebuild is not None:
            design_images = [images[0], prior_image, *images[1:]]
    if format_rebuild is not None:
        revision_info = {"format_issue": format_issue}
    elif prior and not feedback:
        previous_design = prior.get("design", {})
        inputs = {"brief": brief, "copy": copy, "previous_concept": previous_design.get("concept", brief["message"])}
        inputs["previous_art_direction"] = previous_design.get("art_direction", prior.get("scene_prompt"))
        inputs["previous_visual_review"] = prior.get("revision", {}).get("visual_review")
        critique_prompt, critique_images = CRITIQUE, [prior_image, *images]
        fields_to_keep = ("weakness", "hypothesis")
        campaign_viewpoints = {}
        if campaign_intent:
            if mask_model is not None:
                product, source_mask = extract_subject(photo, mask_model, threshold)
                parts = source_parts(source_mask)
            inputs["editor_intent"] = campaign_intent
            critique_prompt += ("\nDesign one replacement campaign from editor_intent. The full composition may change. "
                                "Do not salvage the old subject placement or type merely for continuity. In hypothesis, "
                                "describe a physically visible scene with the actual product. Also return layout_device "
                                "(how positioning, type or a graphic makes that same idea visible) and visible_result "
                                "(what must be apparent in the rendered poster at phone size). Use a layout device "
                                "the editor can draw: straight headline lines, size, selective color, a rectangular or "
                                "elliptical shape, or a small overlap behind the product. Do not plan curved lettering.")
            fields_to_keep += ("layout_device", "visible_result")
            campaign_viewpoints = available_viewpoints()
            if campaign_viewpoints:
                inputs["camera_viewpoints"] = campaign_viewpoints
                critique_prompt += (" If the concept needs a visibly different camera position relative to the source, "
                                    "return viewpoint as one key from camera_viewpoints. This will move the actual camera "
                                    "reference. Return null when the scene does not need a new camera position.")
            if parts is not None:
                critique_prompt += (" If moving one clearly separable piece already visible in the original photo is "
                                    "essential to the hypothesis, return staging_intent describing what moves, where, "
                                    "and why. That piece will actually be moved in the reference without changing its size "
                                    "or count. Use this only where the original exposes the whole visible silhouette. "
                                    "Otherwise return null. Use at most one of viewpoint and staging_intent.")
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
        revision_info = {key: required_text(criticism, key, "완성본 검토") for key in fields_to_keep}
        if parts is not None and criticism.get("staging_intent") is not None:
            staging_intent = required_text(criticism, "staging_intent", "완성본 검토")
            revision_info["staging_intent"] = staging_intent
        elif campaign_viewpoints and (camera_choice := criticism.get("viewpoint")) is not None:
            if not isinstance(camera_choice, str) or camera_choice not in campaign_viewpoints:
                raise ValueError("새 연출에 사용할 촬영 시점을 확인하지 못했습니다")
            viewpoint = camera_choice
    design_input = {"brief": brief, "copy": copy, "fonts": list(settings["fonts"]), "canvas_size": size,
                    "headline_breaks": headline_breaks(copy["headline"])}
    if revision_info:
        design_input.update(brief={"message": brief["message"], "evidence": brief.get("evidence")},
                            idea=idea, revision_feedback=revision_info)
    art_plan = kept.get("art_direction_plan") if proposal or format_rebuild is not None else None
    if art_plan is None and not proposal:
        art_plan = plan_art_direction(ai, fields, design_input["brief"] | {"headline": copy["headline"]}, idea,
                                      images, revision_info)
    if mask_model is not None and source_mask is None:
        product, source_mask = extract_subject(photo, mask_model, threshold)
    group = None
    if format_rebuild is not None and product is not None and prior["design"].get("source_crop") is None:
        parts = source_parts(source_mask)
        if parts is not None:
            group = stage_group(ai, photo, parts, size, prior_image, format_issue, copy)
            if group is None:
                staging = ask_stage(ai, "ad_format_staging", FORMAT_STAGING,
                                    {"target_canvas": size, "format_issue": format_issue, "copy": copy},
                                    [images[0], prior_image, to_float(on_gray(product))])
                if staging.get("move") is not None:
                    staging_intent = required_text(staging, "move", "게시 규격의 상품 배치")
                    revision_info["staging_intent"] = staging_intent
    moved_view = None
    staged_view = None
    moved_piece = None
    if group is not None:
        product, moved_piece, observation = group
        revision_info.update(staged_group=True, staging_observation=observation)
        staged_view = to_float(on_gray(product))
    elif staging_intent:
        if product is None:
            raise ValueError("새 구도에 옮길 상품 영역을 찾지 못했습니다")
        product, moved_piece, observation = stage_subject(ai, photo, source_mask, parts, staging_intent)
        revision_info["staging_observation"] = observation
        staged_view = to_float(on_gray(product))
    elif viewpoint:
        if product is None:
            raise ValueError("새 시점에서 보여 줄 상품 영역을 찾지 못했습니다")
        moved_view = camera_view(photo, source_mask, viewpoint)
    source_framing = proposal.get("source_framing") if proposal else None
    if staged_view is not None:
        source_framing = {"source_crop": None, "reason": revision_info["staging_observation"]}
    elif moved_view is not None:
        source_framing = {"source_crop": None, "reason": "새 촬영 시점에서 상품의 구성을 유지합니다"}
    elif format_rebuild is not None:
        crop = prior["design"].get("source_crop")
        source_framing = {"source_crop": crop, "reason": "채택한 광고의 상품 촬영 범위를 유지합니다"}
    elif not proposal and product is not None:
        framing_input = design_input | {"art_direction_plan": art_plan, "idea": idea}
        if scene_issue:
            framing_input = {"brief": brief, "copy": copy, "idea": idea, "scene_issue": scene_issue,
                             "visible_feature": art_plan["evidence"]}
        source_framing = choose_source_framing(ai, product, framing_input, images)
        source_crop = source_framing["source_crop"]
        scale = "detail" if source_crop is not None else art_plan["shot_scale"]
        if source_crop is None and scale == "detail":
            scale = "close"
        art_plan = art_plan | {"shot_scale": scale}
    if source_framing:
        design_input["source_framing"] = source_framing
    if art_plan:
        design_input["art_direction_plan"] = art_plan | {"layout_guide": THEMES[art_plan["theme"]]["layout"]}

    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as staging:
        result = Path(staging)
        design_task = (DESIGN + "\nUse the selected layout_guide and connect the chosen photo_device to one "
                       "clear reading path. Return a JSON design object.\n" + DESIGN_SCHEMA)
        frame_fields = {}
        if source_framing and not proposal:
            crop = source_framing["source_crop"]
            frame_fields = {"source_crop": crop, "crop": "preserve" if crop is None else "intentional"}
            design_task += "\nSource framing is approved. Use source_framing and lay out the actual view without changing it."
        if format_rebuild is not None:
            design_input["approved_concept"] = prior["design"]["concept"]
            design_task += ("\nThe second image is the selected campaign poster. Keep the existing campaign idea, "
                            "visible product assortment, visual motif and typographic voice. Solve format_issue by "
                            "planning the photograph and type together for canvas_size from the beginning. Think in the "
                            "long axis of the new frame. Use camera depth, the direction of the photographic cue or "
                            "relative subject and headline scale. Do not center the old composition between two "
                            "background-only bands or invent new product components. Treat the first image as identity.")
        elif feedback:
            design_task += ("\nThe second image is the previous poster. Plan the requested_scene_change in revision_feedback "
                            "and preserve unaffected parts of that composition.")
            if scene_issue:
                design_task += " The approved source_framing takes precedence over framing suggested by the editor."
            else:
                design_task += " Do not substitute another change."
        elif revision_info:
            design_task += ("\nMake room for the visual idea in revision_feedback. Give the photography "
                            "and type a new shared composition with the same exact copy.")
            if campaign_intent:
                design_task += (" Use layout_device and show visible_result. Draw the relationship in the actual layout "
                                "through relative scale, overlap or direction; a separate caption above a catalog shot is insufficient.")
        if staged_view is not None:
            design_images = [*design_images, staged_view]
            design_task += ("\nThe final image is the actual rearranged product. Use its new positions and physical "
                            "shape when directing the layout. Gray is transparent.")
            if group is not None:
                design_task += (" Use the full rearranged silhouette as the shape of subject_box. Give its leading "
                                "piece space for the planned visual action and fit the typography into the adjacent "
                                "quiet part of the same scene. Do not flatten the photographed rows back into a line.")
        elif moved_view is not None:
            design_images = [*design_images, to_float(moved_view)]
            design_task += ("\nThe final image shows the requested camera viewpoint with unreconstructed regions. "
                            "Plan space for the visible subject and type around that view.")
        answer = proposal["design"] if proposal else (
            ask_stage(ai, "ad_layout", design_task, design_input, design_images) | frame_fields)
        try:
            layout = design_layers(answer, copy, size, settings["fonts"], result)
            effective = crop_subject(product, answer)
        except ValueError as error:
            if proposal:
                raise
            retry = design_input | {"previous_design": answer, "layout_error": str(error)}
            answer = ask_stage(ai, "ad_layout", design_task, retry, design_images) | frame_fields
            try:
                layout = design_layers(answer, copy, size, settings["fonts"], result)
            except ValueError:
                if format_rebuild is None:
                    raise
                # Start the placement check with known approved lettering when the proposed typography is invalid.
                approved = prior["design"]
                answer = answer | {"boxes": approved["boxes"], "styles": approved["styles"], "accent": None}
                layout = design_layers(answer, copy, size, settings["fonts"], result)
            effective = crop_subject(product, answer)

        composition_reason = proposal.get("composition_reason") if proposal else None
        if product is not None and not proposal:
            answer, layout, composition_reason = review_composition_plan(
                ai, product, answer, layout, result, design_input, images)
            effective = crop_subject(product, answer)
        detail_shot = product is not None and answer.get("source_crop") is not None
        alternate_framing = bool(scene_issue and detail_shot)
        if detail_shot and art_plan:
            art_plan = art_plan | {"shot_scale": "detail"}
            design_input["art_direction_plan"]["shot_scale"] = "detail"
        artwork = layout["artwork"]
        if not proposal:
            direction_input = {"message": brief["message"], "idea": idea, "design": answer}
            angle, scale = ANGLES[art_plan["angle"]], SHOT_SCALES[art_plan["shot_scale"]]
            lighting = art_plan["lighting"]
            direction_input.update(art_direction_plan=art_plan, angle_guide=angle, framing_guide=scale,
                                   lighting_guide=LIGHTING.get(lighting, lighting))
            if source_framing:
                direction_input["source_framing"] = source_framing
            if composition_reason:
                direction_input["approved_composition"] = composition_reason
            if revision_info:
                direction_input.update(revision_info)
            direction_task = PHOTO_DIRECTION + "\nKeep the selected angle, framing, and photo_device in the scene."
            if format_rebuild is not None:
                direction_input["target_photo_aspect"] = [artwork["w"], artwork["h"]]
                direction_input["approved_effect_hint"] = prior.get("revision", {}).get("visible_result")
                direction_task += (" Arrange the subject and lighting for the new composition and art region. "
                                   "Use perspective, directional light or meaningful surroundings across the long axis. "
                                   "Follow the new placement preview and target_photo_aspect. Omit any description "
                                   "of aspect ratio from scene.")
            if detail_shot:
                direction_task += "\nThe approved composition uses a detail crop. Keep that magnification in the photograph."
            direction_images = images[:1]
            if (result / "composition.png").is_file():
                with Image.open(result / "composition.png") as proof:
                    direction_images.append(to_float(proof))
                direction_task += ("\nThe second image is the approved placement before photography. Use the subject "
                                   "position and open regions from it. Its lettering and flat colors are placeholders.")
            if feedback:
                if alternate_framing:
                    direction_task += ("\nUse the requested photographic effect with the approved close framing. "
                                       "Omit any instruction to keep the entire product in view.")
                else:
                    direction_task += "\nDescribe the exact requested_scene_change and omit any conflicting design detail."
            if staged_view is not None:
                direction_images.append(staged_view)
                direction_task += ("\nThe rearrangement guide shows the actual product in its new positions. Keep each "
                                   "piece exactly once there and give all components realistic support on one surface.")
            elif moved_view is not None:
                direction_input["viewpoint"] = viewpoint
                direction_images.append(to_float(moved_view))
                direction_task += ("\nThe last image is the requested viewpoint, with gray gaps to reconstruct. "
                                   "Describe the new scene at that actual angle. Keep the visible product identity and "
                                   "use the planned quiet regions for type. The camera view takes priority over angle_guide.")
            if format_rebuild is not None:
                direction_images.append(prior_image)
                direction_task += (" The last image is the approved poster. Describe the specific use gesture or "
                                   "physical effect visible in its photograph that carries the message, and keep "
                                   "that effect on the same product in the new arrangement. The source photo and "
                                   "neutral placement preview identify the product and geometry; they omit the effect. "
                                   "Do not inherit the approved poster's framing, exact positions, typography or ratio.")
            if campaign_intent:
                direction_task += ("\nLay out a real scene for the hypothesis and the approved headline position. "
                                   "Describe the path, gesture or feature that points toward the planned design. "
                                   "Keep layout_device to guide photography only: include no graphic or text in the photo.")
            direction = ask_stage(ai, "ad_scene_direction", direction_task, direction_input, direction_images)
            answer = answer | {"art_direction": required_text(direction, "scene", "촬영 연출")}

        picture_size = (artwork["w"], artwork["h"])
        subject_box = answer.get("subject_box")
        placement = ("Follow the reference's product placement and scale if it has a plain gray background. "
                     if subject_box else "")
        correction = (f"Required photographic change, taking precedence over other scene details: {feedback} "
                      "Keep all poster copy out of the photograph. " if feedback and not alternate_framing else "")
        guide = (f"{ANGLES[art_plan['angle']]} {SHOT_SCALES[art_plan['shot_scale']]} "
                 if art_plan else "")
        if art_plan and not alternate_framing:
            guide += f"{art_plan['photo_device']} "
        prompt = ("Make a new commissioned photograph with a newly built setting and lighting. "
                  f"{correction}{placement}{guide}Replace the surroundings with the requested setting. "
                  "Keep the product shape, material, visible components and varieties shown in the reference. "
                  "Do not introduce additional products or interchangeable variants. "
                  f"{answer['art_direction']} Preserve the recognizable product details and genuine packaging labels. "
                  "Keep surroundings, props and surfaces free of stray lettering, numbers and logos. "
                  "Add no advertising text or overlays.")
        if answer["edge"] == "blend":
            prompt += f" Keep the outer background low-detail and close to the surrounding {answer['background']} color."
        if answer["crop"] == "preserve":
            prompt += " Keep all of the product in frame with space around its edges."
        if detail_shot:
            prompt += " Keep the same close framing of the product detail shown in the reference."
        if staged_view is not None:
            prompt = ("Make one finished commercial photograph. The first image defines product identity and details. "
                      "The second image defines the new product arrangement, scale and positions. Keep each piece once. "
                      "Do not restore moved pieces to their old positions. Replace gray and repair newly exposed "
                      f"surfaces with physically plausible material, support and shadows. {answer['art_direction']} "
                      "Preserve the actual product and packaging. Add no writing or graphics.")
        elif moved_view is not None:
            prompt = ("Make one finished commercial photograph. The first image shows the original item and its details. "
                      "Use the second image's camera viewpoint and visible arrangement. Gray gaps are missing pixels: "
                      "complete them naturally with details consistent with the first image. Keep the changed angle. "
                      f"{answer['art_direction']} Preserve the actual product, components and packaging. "
                      "Add no writing, graphics or products not specified in the scene.")
        if format_rebuild is not None:
            orientation = "vertical" if picture_size[0] < picture_size[1] else "horizontal"
            if picture_size[0] == picture_size[1]:
                orientation = "square"
            prompt = (f"Compose for this {orientation} photograph, filling its {picture_size[0]} by "
                      f"{picture_size[1]} frame with the planned composition. {prompt}")
        if preview:
            if (result / artwork["file"]).is_file():
                with Image.open(result / artwork["file"]) as planned:
                    scene = planned.convert("RGB")
            else:
                scene = preview_artwork(photo, artwork, size, answer)
            final_review_reason = None
        else:
            model = models["generation"]
            model.use_lora(None, 0)
            generation = settings["generation"]
            render_size = picture_size
            if format_rebuild is not None:
                scale = min(1, sqrt(canvas["width"] * canvas["height"] / (picture_size[0] * picture_size[1])))
                render_size = tuple(round(side * scale) for side in picture_size)
            if staged_view is not None:
                placement_area = subject_box or [8, 8, 84, 84]
                guide = product_reference(photo, effective, render_size, placement_area)
                guide.save(result / "staging.png")
                if moved_piece is not None:
                    piece_box = frame_subject(moved_piece, picture_size, placement_area).getbbox()
                    revision_info["staged_piece_box"] = list(piece_box)
                    on_gray(moved_piece.crop(moved_piece.getbbox())).save(result / "staged-piece.png")
            elif moved_view is not None:
                placement_box = pixel_box(subject_box or [0, 0, 100, 100], picture_size)
                guide = place_camera_view(moved_view, source_mask.getbbox(), photo.size, picture_size, placement_box)
                guide.save(result / "viewpoint.png")
            if staged_view is not None or moved_view is not None:
                original = ImageOps.contain(photo, (640, 640), Image.Resampling.LANCZOS)
                generated = model.edit_with_reference(original, guide, prompt, generation["seed"], generation["steps"])
            else:
                reference = product_reference(photo, effective, render_size, subject_box, answer.get("source_crop"))
                generated = model.edit(reference, prompt, generation["seed"], generation["steps"], size=render_size)
            scene = ImageOps.fit(generated.convert("RGB"), picture_size, Image.Resampling.LANCZOS)
            foreground = find_foreground(scene, models["mask"], threshold)
            if answer["edge"] == "blend":
                scene = blend_edges(scene, foreground, position=(artwork["x"], artwork["y"]), canvas_size=size)
            scene.save(result / artwork["file"])
            Image.fromarray(foreground).save(result / "foreground.png")
            answer, layout, final_review_reason = review_finished(ai, answer, layout, scene, foreground, result,
                                                                  design_input, images[1:],
                                                                  assess_composition=bool(art_plan and not proposal))
        layout.update(fields=fields, brief=brief, design=answer, scene_prompt=prompt, preview=preview,
                      final_review_reason=final_review_reason)
        if art_plan:
            layout["art_direction_plan"] = art_plan
        if source_framing:
            layout["source_framing"] = source_framing
        if composition_reason:
            layout["composition_reason"] = composition_reason
        if preview:
            scene.save(result / "scene.png")
        poster = compose(layout, result)
        poster.save(result / "ad.png")
        if revision_info:
            layout["revision"] = revision_info
        if viewpoint:
            layout["viewpoint"] = viewpoint
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
