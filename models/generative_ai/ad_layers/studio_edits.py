"""완성 시안에서 지정한 레이어나 사진 영역만 고칩니다."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .creative_brief import TEXT_ROLES, required_text, validate_boxes
from .layers import compose, load_layout, load_settings, save_layout
from .scene_ad import check_text_overlap, layout_texts, pixel_box
from .text_render import render_texts


def foreground_mask(folder, layout):
    artwork = layout["artwork"]
    with Image.open(folder / "foreground.png") as mask:
        if mask.size != (artwork["w"], artwork["h"]):
            raise ValueError("피사체 마스크 크기가 사진과 다릅니다")
        return np.asarray(mask.convert("L")) > 128


def change_text(stage, layout, changes):
    if not isinstance(changes, dict) or not changes or set(changes) - set(TEXT_ROLES):
        raise ValueError("제목, 상품명, 가격에서 고칠 글자 레이어를 지정하세요")
    design = layout["design"]
    boxes, styles = design["boxes"].copy(), design["styles"].copy()
    copy = {text["role"]: text["text"] for text in layout["texts"]}
    for role, change in changes.items():
        if (not isinstance(change, dict) or not change
                or change.keys() - {"box", "font", "align", "color", "size", "max_lines", "first_line"}):
            raise ValueError("글자 역할 아래에 box, font, align, color, size, max_lines, first_line만 입력하세요")
        if "box" in change:
            boxes[role] = change["box"]
        style = styles[role] | {key: value for key, value in change.items() if key not in ("box", "first_line")}
        if "first_line" in change or style["max_lines"] == 1:
            style.pop("break_after", None)
        if (first_line := change.get("first_line")) is not None:
            if not isinstance(first_line, str):
                raise ValueError("첫 줄에 배치할 실제 문구를 입력하세요")
            words, prefix = copy[role].split(), first_line.split()
            if not prefix or len(prefix) >= len(words) or words[:len(prefix)] != prefix:
                raise ValueError("첫 줄은 기존 문구의 앞부분과 같아야 합니다")
            style.update(max_lines=2, break_after=len(prefix))
        styles[role] = style
    revised = design | {"boxes": boxes, "styles": styles}
    texts = layout_texts(revised, copy, layout["size"], load_settings()["fonts"])
    render_texts([text for text in texts if text["role"] in changes], tuple(layout["size"]), stage)
    layout.update(design=revised, texts=texts)


def change_region(stage, layout, arguments, generation):
    validate_boxes({"area": arguments.get("box")}, ("area",))
    instruction = required_text(arguments, "instruction", "영역 편집")
    x, y, width, height = pixel_box(arguments["box"], layout["size"])
    artwork = layout["artwork"]
    x, y = x - artwork["x"], y - artwork["y"]
    if not (0 <= x < x + width <= artwork["w"] and 0 <= y < y + height <= artwork["h"]):
        raise ValueError("영역 편집은 사진 안에서만 할 수 있습니다")
    if foreground_mask(stage, layout)[y:y + height, x:x + width].any():
        raise ValueError("영역 편집이 제품에 닿습니다. 제품 바깥을 다시 선택하세요")
    path = stage / artwork["file"]
    with Image.open(path) as photo:
        scene = photo.convert("RGBA")
    generation = generation()
    model = generation["generation"]
    model.use_lora(None, 0)
    settings = load_settings()["generation"]
    bounds = (x, y, x + width, y + height)
    relative = tuple(round(100 * value / limit) for value, limit in zip(bounds, (*scene.size, *scene.size), strict=True))
    prompt = (f"Edit the scenery only inside the target region at percentages {relative} (left, top, right, bottom). "
              f"{instruction} Match the original lighting, texture, camera, and framing. Add no lettering.")
    candidate = model.edit(scene.convert("RGB"), prompt, settings["seed"], settings["steps"], size=scene.size)
    candidate = ImageOps.fit(candidate.convert("RGBA"), scene.size, Image.Resampling.LANCZOS)
    margin = min(width, height) // 12
    blend = Image.new("L", (width, height))
    ImageDraw.Draw(blend).rectangle((margin, margin, width - margin - 1, height - margin - 1), fill=255)
    if margin:
        blend = blend.filter(ImageFilter.GaussianBlur(margin / 2))
    # 사진 외곽을 합성할 때 쓰는 기존 투명도는 유지합니다.
    alpha = scene.getchannel("A")
    scene.paste(candidate.crop(bounds), (x, y), blend)
    scene.putalpha(alpha)
    scene.save(path)


def edit_version(source, output, action, arguments, generation):
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary)
        shutil.copytree(source, stage, dirs_exist_ok=True)
        layout = load_layout(stage)
        if action == "edit_text":
            change_text(stage, layout, arguments.get("changes"))
        elif action == "edit_region":
            change_region(stage, layout, arguments, generation)
        else:
            raise ValueError("지원하지 않는 편집 작업입니다")
        check_text_overlap(foreground_mask(stage, layout), layout, stage)
        (stage / "before-typesetting.png").unlink(missing_ok=True)
        compose(layout, stage).save(stage / "ad.png")
        save_layout(layout, stage)
        stage.rename(output)
