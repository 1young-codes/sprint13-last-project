"""그래픽 카드 없이 도는 레이어 계산: 제품 놓기, 접지 그림자, 글자 배치, 레이어 합치기.

레이어 정보(layout.json)는 아래 모양입니다. 위치는 화소, 파일은 같은 폴더의 투명 PNG입니다.
    {"size": [가로, 세로], "mood": "minimal",
     "background": {"file": "background.png"},
     "product": {"file": "product.png", "x": 0, "y": 0},
     "shadow": {"file": "shadow.png", "dx": 0, "dy": 0},      # 제품 위치 기준. 제품을 옮기면 따라감
     "texts": [{"role": "title", "text": "...", "font": "...", "color": "#rrggbb",
                "x": 0, "y": 0, "w": 0, "h": 0, "file": "text_1_title.png"}]}
"""

import json
import tomllib
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]


@cache
def load_settings() -> dict:
    with open(PACKAGE_DIR / "settings.toml", "rb") as file:
        return tomllib.load(file)


def to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), np.float32) / 255


def cutout_product(photo: Image.Image, mask: np.ndarray, threshold: float) -> Image.Image:
    """제품 주변의 배경을 제거한 사진을 만듭니다."""
    alpha = Image.fromarray((mask.clip(0, 1) * 255).astype(np.uint8))
    box = (mask > threshold).nonzero()
    if not len(box[0]):
        raise ValueError("사진에서 제품을 찾지 못했습니다")
    crop = (box[1].min(), box[0].min(), box[1].max() + 1, box[0].max() + 1)
    product = photo.convert("RGB").crop(crop)
    product.putalpha(alpha.crop(crop))
    return product


def place_product(photo: Image.Image, mask: np.ndarray, size: tuple[int, int], rule: dict) -> tuple[Image.Image, int, int]:
    """사진에서 제품만 투명 PNG로 잘라 크기를 맞추고, 캔버스에 놓을 왼쪽 위 좌표를 정합니다."""
    product = cutout_product(photo, mask, rule["mask_threshold"])
    width, height = size
    scale = min(rule["max_share"] * width / product.width, rule["max_share"] * height / product.height)
    product = product.resize((max(1, round(product.width * scale)), max(1, round(product.height * scale))),
                             Image.Resampling.LANCZOS)
    x = round(rule["center_x"] * width - product.width / 2)
    y = round(rule["bottom_y"] * height - product.height)
    return product, x, y


def contact_shadow(product: Image.Image, overhead: bool) -> tuple[Image.Image, int, int]:
    """제품 마스크에서 바닥에 닿는 윤곽을 읽어 부드러운 그림자를 만듭니다."""
    width, height = product.size
    unit = max(width, height)
    pad = max(1, round(unit * 0.12))
    footprint = product.getchannel("A")
    if not overhead:
        # 앞뒤로 놓인 물건은 같은 사진에서도 바닥에 닿는 높이가 다릅니다.
        coverage = np.asarray(footprint)
        rows = np.arange(height)[:, None]
        bottom = np.where(coverage > 128, rows, -height).max(axis=0)
        depth = max(1, round(unit * 0.045))
        near_bottom = (rows > bottom - depth) & (rows <= bottom + 1)
        footprint = Image.fromarray(np.where(near_bottom, coverage, 0).astype(np.uint8))
    alpha = Image.new("L", (width + 2 * pad, height + 2 * pad))
    alpha.paste(footprint, (pad, pad))
    tight = alpha.filter(ImageFilter.GaussianBlur(max(1, unit * 0.008))).point(lambda v: round(v * 0.38))
    soft = alpha.filter(ImageFilter.GaussianBlur(max(1, unit * 0.025))).point(lambda v: round(v * 0.20))
    shadow = Image.new("RGBA", alpha.size)
    shadow.putalpha(ImageChops.screen(tight, soft))
    offset = round(unit * 0.012) - pad
    box = shadow.getbbox()
    if box is None:
        return shadow, offset, offset
    return shadow.crop(box), offset + box[0], offset + box[1]


def template_boxes(base: np.ndarray, rule: list[dict]) -> list[dict]:
    """기본 글자 배치에 배경 밝기에 맞는 색을 지정합니다."""
    height, width = base.shape[:2]
    boxes = []
    for slot in rule:
        x, y, w, h = (round(v * size) for v, size in zip(slot["box"], (width, height, width, height)))
        light = base[y:y + h, x:x + w].mean() > 0.55
        boxes.append({"x": x, "y": y, "w": w, "h": h, "role": slot["role"],
                      "color": "#1a1a1a" if light else "#ffffff"})
    return boxes


def compose(layout: dict, folder: Path) -> Image.Image:
    """layout.json대로 레이어를 아래부터 합칩니다. 제품을 옮기면 그림자가 같이 옮겨집니다."""
    canvas = Image.open(folder / layout["background"]["file"]).convert("RGBA")
    artwork = layout.get("artwork")
    if artwork:
        paste_clipped(canvas, Image.open(folder / artwork["file"]).convert("RGBA"), artwork["x"], artwork["y"])
    graphic = layout.get("graphic")
    foreground = None

    def paste_design_layer(info):
        nonlocal foreground
        with Image.open(folder / info["file"]) as source:
            layer = source.convert("RGBA")
        if info.get("depth") == "behind_subject" and not layout.get("preview"):
            if foreground is None:
                with Image.open(folder / "foreground.png") as mask:
                    foreground = mask.convert("L").point(lambda value: 255 if value > 128 else 0)
            x, y = info["x"] - artwork["x"], info["y"] - artwork["y"]
            visible = ImageChops.invert(foreground.crop((x, y, x + layer.width, y + layer.height)))
            alpha = ImageChops.multiply(layer.getchannel("A"), visible)
            if info is graphic:
                left, top = info["x"], info["y"]
                light = canvas.crop((left, top, left + layer.width, top + layer.height)).convert("L")
                samples = np.asarray(light)[np.asarray(alpha) > 0]
                if samples.size and (base_light := np.percentile(samples, 90)) > 0:
                    # 원래 사진의 접지 그림자가 색면 경계에서 끊기지 않게 합니다.
                    shade = light.point(lambda value: min(255, round(255 * value / base_light)))
                    layer = ImageChops.multiply(layer.convert("RGB"), shade.convert("RGB"))
            layer.putalpha(alpha)
        paste_clipped(canvas, layer, info["x"], info["y"])

    if graphic:
        paste_design_layer(graphic)
    product = layout.get("product")
    if product:
        shadow = layout.get("shadow")
        if shadow:
            paste_clipped(canvas, Image.open(folder / shadow["file"]).convert("RGBA"),
                          product["x"] + shadow["dx"], product["y"] + shadow["dy"])
        paste_clipped(canvas, Image.open(folder / product["file"]).convert("RGBA"), product["x"], product["y"])
    for text in layout["texts"]:
        for line in text.get("lines", (text,)):
            paste_design_layer(line)
    return canvas.convert("RGB")


def paste_clipped(canvas: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    """alpha_composite는 캔버스 밖 좌표를 받지 않으므로, 캔버스 안에 들어오는 부분만 잘라 얹습니다."""
    left, top = max(0, -x), max(0, -y)
    right, bottom = min(layer.width, canvas.width - x), min(layer.height, canvas.height - y)
    if right > left and bottom > top:
        canvas.alpha_composite(layer.crop((left, top, right, bottom)), (x + left, y + top))


def save_layout(layout: dict, folder: Path) -> None:
    (folder / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")


def load_layout(folder: Path) -> dict:
    return json.loads((folder / "layout.json").read_text(encoding="utf-8"))
