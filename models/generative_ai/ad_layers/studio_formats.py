"""채택한 광고의 사진을 보호하면서 게시 규격마다 글자를 다시 배치합니다."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .creative_brief import ask_stage
from .layers import compose, load_layout, load_settings, save_layout, to_float
from .scene_ad import DESIGN_SCHEMA, check_text_overlap, design_layers
from .studio_edits import foreground_mask


FORMATS = {"1:1": (1024, 1024), "4:5": (864, 1080), "9:16": (1080, 1920), "16:9": (1280, 720),
           "thumbnail": (600, 600)}

REFRAME = """Reframe the approved poster's exact typography for a new aspect ratio. The first image is the
approved poster and the second is the actual photo in the new frame. Keep the new photo unchanged. Use its
quiet space and the product bounds to place the exact headline, product name and price. Keep the approved
campaign's colors and hierarchy. Make each text size appropriate for the new width. If copy needs two lines,
use break_after at a natural Korean phrase boundary and keep quantities and their counters together. Return only a JSON object
with boxes, styles and accent (an optional overlay, or null). Repair any reported overlap or fit error."""


def extend_edges(photo, size):
    contained = ImageOps.contain(photo, size, Image.Resampling.LANCZOS)
    left, top = (size[0] - contained.width) // 2, (size[1] - contained.height) // 2
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
    return frame


def frame_photo(photo, mask, size):
    width, height = photo.size
    target_width, target_height = size
    ratio = target_width / target_height
    crop_width, crop_height = min(width, round(height * ratio)), min(height, round(width / ratio))
    left, top, right, bottom = mask.getbbox()
    if right - left <= crop_width and bottom - top <= crop_height:
        x = min(left, width - crop_width, max(0, right - crop_width, (width - crop_width) // 2))
        y = min(top, height - crop_height, max(0, bottom - crop_height, (height - crop_height) // 2))
        bounds = (x, y, x + crop_width, y + crop_height)
        return (photo.crop(bounds).resize(size, Image.Resampling.LANCZOS),
                mask.crop(bounds).resize(size, Image.Resampling.NEAREST))
    return extend_edges(photo, size), ImageOps.pad(mask, size, Image.Resampling.NEAREST, color=0)


def adapt_format(source, output, format_name, ai):
    layout = load_layout(source)
    size = FORMATS[format_name]
    threshold_mask = foreground_mask(source, layout)
    mask = Image.fromarray(threshold_mask.astype(np.uint8) * 255)
    if not mask.getbbox():
        raise ValueError("크기 변경에 사용할 제품 영역이 없습니다")
    with Image.open(source / layout["artwork"]["file"]) as original:
        photo = original.convert("RGB")
    design = layout["design"]
    framing, framing_mask = frame_photo(photo, mask, size)
    with Image.open(source / "ad.png") as poster:
        images = [to_float(poster), to_float(framing)]
    copy = {item["role"]: item["text"] for item in layout["texts"]}
    fonts = load_settings()["fonts"]
    bounds = [round(100 * point / side) for point, side in zip(framing_mask.getbbox(), (*size, *size), strict=True)]
    inputs = {"size": size, "copy": copy, "fonts": list(fonts), "approved_design": design, "product_bounds": bounds}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary)
        for _ in range(2):
            answer = ask_stage(ai, "studio_reframe", REFRAME + "\n" + DESIGN_SCHEMA, inputs, images)
            revised = design | {"artwork_box": [0, 0, 100, 100], "boxes": answer.get("boxes"),
                                "styles": answer.get("styles"), "accent": answer.get("accent")}
            revised.pop("subject_box", None)
            try:
                candidate = design_layers(revised, copy, size, fonts, stage)
                check_text_overlap(np.asarray(framing_mask) > 128, candidate, stage)
            except ValueError as error:
                inputs["layout_error"] = str(error)
                continue
            framing.save(stage / candidate["artwork"]["file"])
            framing_mask.save(stage / "foreground.png")
            compose(candidate, stage).save(stage / "ad.png")
            candidate.update(fields=layout["fields"], brief=layout["brief"], design=revised,
                             scene_prompt=layout["scene_prompt"], format=format_name)
            save_layout(candidate, stage)
            stage.rename(output)
            return
    raise ValueError(inputs["layout_error"])
