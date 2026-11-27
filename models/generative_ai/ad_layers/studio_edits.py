"""완성 시안에서 지정한 레이어나 사진 영역만 고칩니다."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .creative_brief import TEXT_ROLES, ask_stage, required_text, validate_boxes
from .layers import compose, load_layout, load_settings, save_layout, to_float
from .scene_ad import check_text_overlap, layout_texts, pixel_box, review_finished
from .staging import padded_box, staged_evidence
from .text_render import render_texts


def foreground_mask(folder, layout):
    artwork = layout["artwork"]
    with Image.open(folder / "foreground.png") as mask:
        if mask.size != (artwork["w"], artwork["h"]):
            raise ValueError("피사체 마스크 크기가 사진과 다릅니다")
        return np.asarray(mask.convert("L")) > 128


def line_break_after(text, first_line):
    if not isinstance(first_line, str):
        raise ValueError("첫 줄에 배치할 실제 문구를 입력하세요")
    words, prefix = text.split(), first_line.split()
    if not prefix or len(prefix) >= len(words) or words[:len(prefix)] != prefix:
        raise ValueError("첫 줄은 기존 문구의 앞부분과 같아야 합니다")
    return len(prefix)


def change_text(stage, layout, changes):
    if not isinstance(changes, dict) or not changes or set(changes) - set(TEXT_ROLES):
        raise ValueError("제목, 상품명, 가격에서 고칠 글자 레이어를 지정하세요")
    design = layout["design"]
    boxes, styles = design["boxes"].copy(), design["styles"].copy()
    copy = {text["role"]: text["text"] for text in layout["texts"]}
    for role, change in changes.items():
        if (not isinstance(change, dict) or not change
                or change.keys() - {"box", "font", "align", "color", "size", "max_lines", "first_line", "depth", "emphasis", "lines"}):
            raise ValueError("지원되는 글자 배치와 스타일을 지정해야 합니다")
        if "box" in change:
            boxes[role] = change["box"]
        style = styles[role] | {key: value for key, value in change.items() if key not in ("box", "first_line")}
        if "first_line" in change or style["max_lines"] == 1:
            style.pop("break_after", None)
        if (first_line := change.get("first_line")) is not None:
            style.update(max_lines=2, break_after=line_break_after(copy[role], first_line))
        if "lines" not in change and "break_after" not in style:
            style.pop("lines", None)
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
    paste_region(scene, candidate.crop(bounds), bounds)
    scene.save(path)


def paste_region(scene, patch, bounds):
    margin = min(patch.size) // 12
    blend = Image.new("L", patch.size)
    ImageDraw.Draw(blend).rectangle((margin, margin, patch.width - margin - 1, patch.height - margin - 1), fill=255)
    if margin:
        blend = blend.filter(ImageFilter.GaussianBlur(margin / 2))
    # 사진 외곽을 합성할 때 쓰는 기존 투명도는 유지합니다.
    alpha = scene.getchannel("A")
    scene.paste(patch, bounds[:2], blend)
    scene.putalpha(alpha)


REPAIR_DETAIL = """Plan one local photo retouch. Image one is the real item. Image two shows the photographed
area around the same item. Image three shows the whole new scene. Identify the tiny visible detail described
by the editor, as it exists in the original, and where it belongs in image two. The target box must cover both
the old defect and the expected footprint and contact of the restored part. Include any connected effect that
needs to follow the new part, and leave edge room for blending. Keep unrelated contours and shadows. Return
one JSON object with source_box, target_box, and instruction. Both boxes are JSON arrays of exactly four
integers [left, top, width, height] in 0–100 percentages, relative to image one and image two respectively.
Instruction describes the correct construction, its placement, and how it appears in the current action and
lighting, in English. Set target_box to null if this needs more than a local retouch."""


def repair_staged_detail(stage, layout, arguments, generation, ai):
    issue = required_text(arguments, "instruction", "상품 일부 복원")
    path = stage / layout["artwork"]["file"]
    with Image.open(path) as image:
        scene = image.convert("RGBA")
    original, visible = staged_evidence(stage, layout, scene)
    detail = scene.crop(visible)
    data = {"issue": issue}
    images = list(map(to_float, (original, detail, scene)))
    for _ in range(2):
        plan = ask_stage(ai, "studio_repair_detail", REPAIR_DETAIL, data, images)
        if "target_box" in plan and plan["target_box"] is None:
            raise ValueError("상품의 이 차이는 사진 일부만 고쳐 해결할 수 없습니다")
        try:
            for role in ("source_box", "target_box"):
                validate_boxes({role: plan.get(role)}, (role,))
            instruction = required_text(plan, "instruction", "상품 일부 복원")
            break
        except ValueError as error:
            data.update(previous_plan=plan, correction=str(error))
    else:
        raise ValueError(data["correction"])
    sx, sy, sw, sh = pixel_box(plan["source_box"], original.size)
    tx, ty, w, h = pixel_box(plan["target_box"], detail.size)
    if min(sw, sh, w, h) == 0:
        raise ValueError("원본에서 확인되는 작은 상품 영역만 복원할 수 있습니다")
    x, y = visible[0] + tx, visible[1] + ty
    piece = layout["revision"]["staged_piece_box"]
    short_side = min(piece[2] - piece[0], piece[3] - piece[1])
    x, y, x1, y1 = padded_box((x, y, x + w, y + h), max(2, short_side // 18), scene.size)
    target = (x, y, x1, y1)
    if (x1 - x) * (y1 - y) > detail.width * detail.height // 3:
        raise ValueError("원본에서 확인되는 작은 상품 영역만 복원할 수 있습니다")
    left, top, right, bottom = padded_box(target, max(w, h, short_side // 2), scene.size)
    window = (left, top, right, bottom)
    context = ImageOps.contain(scene.crop(window).convert("RGB"), (576, 576))
    source_window = padded_box((sx, sy, sx + sw, sy + sh), min(original.size) // 3, original.size)
    reference = ImageOps.contain(original.crop(source_window), (320, 320))
    model = generation()["generation"]
    model.use_lora(None, 0)
    settings = load_settings()["generation"]
    region = tuple(round(100 * edge / length) for edge, length in zip(
        (x - left, y - top, x1 - left, y1 - top), (right - left, bottom - top) * 2, strict=True))
    prompt = ("The first image contains the authentic working detail. Retouch only the target region of the second "
              f"image at percentages {region} (left, top, right, bottom). {instruction} "
              "Preserve its angle, framing, surroundings and lighting. Add no lettering.")
    candidate = model.edit_with_reference(reference, context, prompt, settings["seed"], settings["steps"])
    patch = candidate.convert("RGBA").resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    paste_region(scene, patch.crop((x - left, y - top, x1 - left, y1 - top)), target)
    scene.save(path)


def change_composition(stage, layout, arguments, ai, references):
    issue = required_text(arguments, "instruction", "배치 편집")
    copy = {text["role"]: text["text"] for text in layout["texts"]}
    inputs = {"copy": copy, "fonts": list(load_settings()["fonts"]), "composition_issue": issue}
    with Image.open(stage / layout["artwork"]["file"]) as photo:
        scene = photo.copy()
    design, revised, reason = review_finished(ai, layout["design"], layout, scene, foreground_mask(stage, layout),
                                             stage, inputs, references, assess_composition=True)
    if "graphic" not in revised:
        layout.pop("graphic", None)
    layout.update(revised, design=design, final_review_reason=reason)


def edit_version(source, output, action, arguments, generation, *, ai=None, references=()):
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary)
        shutil.copytree(source, stage, dirs_exist_ok=True)
        layout = load_layout(stage)
        if action == "edit_text":
            change_text(stage, layout, arguments.get("changes"))
        elif action == "edit_composition" and ai is not None:
            change_composition(stage, layout, arguments, ai, references)
        elif action == "edit_region":
            change_region(stage, layout, arguments, generation)
        elif action == "repair_staged_detail" and ai is not None:
            repair_staged_detail(stage, layout, arguments, generation, ai)
        else:
            raise ValueError("지원하지 않는 편집 작업입니다")
        check_text_overlap(foreground_mask(stage, layout), layout, stage)
        if action != "edit_composition":
            (stage / "before-typesetting.png").unlink(missing_ok=True)
        compose(layout, stage).save(stage / "ad.png")
        save_layout(layout, stage)
        stage.rename(output)
