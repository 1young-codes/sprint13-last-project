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


def blurred(values: np.ndarray, radius: int) -> np.ndarray:
    from ..camera_angle_presets.fill_empty_area import box_blur

    for _ in range(3):  # 상자 평균 세 번 ≈ 가우시안 흐림
        values = box_blur(values, radius)
    return values


def shadow_layer(scene: np.ndarray, empty: np.ndarray, region: np.ndarray, product_mask: np.ndarray,
                 rule: dict) -> Image.Image:
    """제품이 있는 장면과 빈 배경의 밝기 차이로 그림자(검정)와 반사·빛(흰색) 레이어를 만듭니다.
    빈 배경 위에 이 레이어를 얹으면 장면의 밝기가 돌아옵니다: 검정은 bg·(1-a), 흰색은 bg + (1-bg)·a.

    둘 다 흐린 뒤 비교합니다. 빈 배경을 새로 채우면 나무 결 같은 잔무늬가 달라지는데, 그건 그림자가 아닙니다.
    색이 달라진 곳도 뺍니다. 그림자와 빛은 세 색을 거의 같은 비율로 바꾸지만, 생성 모델이 제품 옆에
    더 그린 물건(예: 딸기 하나 더)은 색이 다릅니다. 그런 물건을 밝기만 남기면 검은 덩어리가 됩니다."""
    radius = max(1, round(rule["blur"] * scene.shape[1] / 3))
    scene, empty = blurred(scene, radius), blurred(empty, radius)
    weights = np.array([0.299, 0.587, 0.114], np.float32)
    lit, bare = scene @ weights, empty @ weights
    darker = np.clip((bare - lit) / np.maximum(bare, 1e-3), 0, 1)
    brighter = np.clip((lit - bare) / np.maximum(1 - bare, 1e-3), 0, 1)
    ratio = (scene + 1e-3) / (empty + 1e-3)
    tinted = (ratio.max(axis=2) - ratio.min(axis=2)) / ratio.mean(axis=2) >= rule["max_tint"]
    neutral = ~grown(tinted, 3 * radius)  # 흐림 때문에 물건 가장자리는 색이 옅어지므로 그만큼 더 뺍니다
    # 다시 채운 둘레는 전체 밝기도 장면과 조금 다릅니다. 둘레 끝에서 뚝 끊기면 경계선이 보이므로 끝으로 갈수록 옅게 합니다
    fade = np.clip(ndimage.distance_transform_edt(region) / max(1.0, rule["fade"] * scene.shape[1]), 0, 1)
    alpha = np.where(lit < bare, darker, brighter) * fade * neutral * (1 - product_mask)
    alpha[alpha < rule["min_alpha"]] = 0
    color = np.where(lit < bare, 0, 255).astype(np.uint8)
    rgba = np.dstack([color, color, color, (alpha * 255).round().astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def find_text_boxes(draft: np.ndarray, base: np.ndarray, product_mask: np.ndarray, rule: dict) -> list[dict]:
    """초안(base 위에 모델이 글자를 그린 그림)과 base의 차이로 글자 줄 자리와 색을 찾습니다. 큰 것부터 돌려줍니다."""
    height, width = base.shape[:2]
    # 생성 모델은 글자만 더하지 않고 그림 전체를 조금씩 다시 그립니다(색, 밝기, 나무 결).
    # 넓게 번지는 차이는 흐려서 빼고, 글자처럼 좁고 선명한 차이만 남깁니다.
    # 번지는 차이를 잴 때 글자처럼 크게 바뀐 화소는 빼고 잽니다. 넣으면 글자 둘레까지 차이가 번져 상자가 커집니다.
    change = draft - base
    calm = (np.abs(change).max(axis=2) <= rule["diff_threshold"]).astype(np.float32)
    radius = max(1, round(rule["drift_blur"] * width / 3))
    drift = blurred(change * calm[..., None], radius) / np.maximum(blurred(calm, radius), 1e-3)[..., None]
    difference = np.abs(change - drift).max(axis=2)
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
        if y1 - y0 > rule["max_height"] * height or (x1 - x0) * (y1 - y0) > rule["max_box_area"] * width * height:
            continue  # 글자 한 줄로 보기엔 너무 큰 영역: 바닥 무늬나 물건이 통째로 바뀐 곳
        # 글자 색: 바뀐 화소 중 가장 많이 바뀐 30%의 중앙값 (가장자리 흐림과 배경이 섞인 화소를 뺌)
        strength = difference[area][inside]
        colors = draft[area][inside][strength >= np.quantile(strength, 0.7)]
        color = "#" + "".join(f"{round(v * 255):02x}" for v in np.median(colors, axis=0))
        boxes.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "color": color})
    boxes.sort(key=lambda b: b["w"] * b["h"], reverse=True)
    return boxes[:rule["max_boxes"]]


def template_boxes(base: np.ndarray, rule: list[dict]) -> list[dict]:
    """생성 모델이 초안에 글자를 그리지 않았을 때 쓰는 기본 배치 틀. 글자 색은 그 자리 배경이 밝으면 검정, 어두우면 흰색."""
    height, width = base.shape[:2]
    boxes = []
    for slot in rule:
        x, y, w, h = (round(v * size) for v, size in zip(slot["box"], (width, height, width, height)))
        light = base[y:y + h, x:x + w].mean() > 0.55
        boxes.append({"x": x, "y": y, "w": w, "h": h, "color": "#1a1a1a" if light else "#ffffff"})
    return boxes


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
