"""그래픽 카드 없이 도는 레이어 계산: 제품 놓기, 그림자 분리, 초안에서 글자 자리 찾기, 레이어 합치기.

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
from PIL import Image
from scipy import ndimage

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]


@cache
def load_settings() -> dict:
    with open(PACKAGE_DIR / "settings.toml", "rb") as file:
        return tomllib.load(file)


def to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), np.float32) / 255


def place_product(photo: Image.Image, mask: np.ndarray, size: tuple[int, int], rule: dict) -> tuple[Image.Image, int, int]:
    """사진에서 제품만 투명 PNG로 잘라 크기를 맞추고, 캔버스에 놓을 왼쪽 위 좌표를 정합니다."""
    alpha = Image.fromarray((mask.clip(0, 1) * 255).astype(np.uint8))
    box = (mask > rule["mask_threshold"]).nonzero()
    if not len(box[0]):
        raise ValueError("사진에서 제품을 찾지 못했습니다")
    crop = (box[1].min(), box[0].min(), box[1].max() + 1, box[0].max() + 1)
    product = photo.convert("RGB").crop(crop)
    product.putalpha(alpha.crop(crop))
    width, height = size
    scale = min(rule["max_share"] * width / product.width, rule["max_share"] * height / product.height)
    product = product.resize((max(1, round(product.width * scale)), max(1, round(product.height * scale))),
                             Image.Resampling.LANCZOS)
    x = round(rule["center_x"] * width - product.width / 2)
    y = round(rule["bottom_y"] * height - product.height)
    return product, x, y


def product_canvas_mask(product: Image.Image, x: int, y: int, size: tuple[int, int]) -> np.ndarray:
    """캔버스 크기의 제품 마스크 (0~1)."""
    canvas = Image.new("L", size, 0)
    canvas.paste(product.getchannel("A"), (x, y))
    return np.asarray(canvas, np.float32) / 255


def shadow_region(mask: np.ndarray, reach: float, threshold: float) -> np.ndarray:
    """제품과 그 주변(제품 크기 × reach): 빈 배경을 만들 때 지우고, 그림자를 찾는 곳입니다."""
    solid = mask > threshold
    rows, cols = solid.nonzero()
    grow = max(1, round(reach * max(np.ptp(rows) + 1, np.ptp(cols) + 1)))
    return grown(solid, grow)


def grown(solid: np.ndarray, distance: int) -> np.ndarray:
    """solid를 distance 화소만큼 넓힙니다 (넓히기를 여러 번 반복하는 것보다 훨씬 빠름)."""
    return ndimage.distance_transform_edt(~solid) <= distance


def shadow_layer(scene: np.ndarray, empty: np.ndarray, region: np.ndarray, product_mask: np.ndarray,
                 min_alpha: float) -> Image.Image:
    """제품이 있는 장면과 빈 배경의 밝기 차이로 그림자(검정)와 반사·빛(흰색) 레이어를 만듭니다.
    빈 배경 위에 이 레이어를 얹으면 장면의 밝기가 그대로 돌아옵니다: 검정은 bg·(1-a), 흰색은 bg + (1-bg)·a."""
    weights = np.array([0.299, 0.587, 0.114], np.float32)
    lit, bare = scene @ weights, empty @ weights
    darker = np.clip((bare - lit) / np.maximum(bare, 1e-3), 0, 1)
    brighter = np.clip((lit - bare) / np.maximum(1 - bare, 1e-3), 0, 1)
    alpha = np.where(lit < bare, darker, brighter) * region * (1 - product_mask)
    alpha[alpha < min_alpha] = 0
    color = np.where(lit < bare, 0, 255).astype(np.uint8)
    rgba = np.dstack([color, color, color, (alpha * 255).round().astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def find_text_boxes(draft: np.ndarray, base: np.ndarray, product_mask: np.ndarray, rule: dict) -> list[dict]:
    """초안(base 위에 모델이 글자를 그린 그림)과 base의 차이로 글자 줄 자리와 색을 찾습니다. 큰 것부터 돌려줍니다."""
    height, width = base.shape[:2]
    difference = np.abs(draft - base).max(axis=2)
    keep_out = grown(product_mask > 0.5, max(1, round(rule["product_margin"] * width)))
    changed = (difference > rule["diff_threshold"]) & ~keep_out
    reach = np.ones((max(1, round(rule["merge_y"] * height)) * 2 + 1, max(1, round(rule["merge_x"] * width)) * 2 + 1), bool)
    labels, _ = ndimage.label(ndimage.binary_dilation(changed, structure=reach))
    boxes = []
    for index, area in enumerate(ndimage.find_objects(labels), start=1):
        inside = changed[area] & (labels[area] == index)
        if inside.sum() < rule["min_area"] * width * height:
            continue
        rows, cols = inside.nonzero()
        y0, x0 = area[0].start + rows.min(), area[1].start + cols.min()
        y1, x1 = area[0].start + rows.max() + 1, area[1].start + cols.max() + 1
        # 글자 색: 바뀐 화소 중 가장 많이 바뀐 30%의 중앙값 (가장자리 흐림과 배경이 섞인 화소를 뺌)
        strength = difference[area][inside]
        colors = draft[area][inside][strength >= np.quantile(strength, 0.7)]
        color = "#" + "".join(f"{round(v * 255):02x}" for v in np.median(colors, axis=0))
        boxes.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "color": color})
    boxes.sort(key=lambda b: b["w"] * b["h"], reverse=True)
    return boxes[:rule["max_boxes"]]


def compose(layout: dict, folder: Path) -> Image.Image:
    """layout.json대로 레이어를 아래부터 합칩니다. 제품을 옮기면 그림자가 같이 옮겨집니다."""
    canvas = Image.open(folder / layout["background"]["file"]).convert("RGBA")
    product = layout["product"]
    shadow = layout.get("shadow")
    if shadow:
        paste_clipped(canvas, Image.open(folder / shadow["file"]).convert("RGBA"),
                      product["x"] + shadow["dx"], product["y"] + shadow["dy"])
    paste_clipped(canvas, Image.open(folder / product["file"]).convert("RGBA"), product["x"], product["y"])
    for text in layout["texts"]:
        paste_clipped(canvas, Image.open(folder / text["file"]).convert("RGBA"), text["x"], text["y"])
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
