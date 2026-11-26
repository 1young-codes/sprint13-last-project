"""그래픽 카드와 모델 없이 도는 확인: 제품 놓기, 그림자 분리, 글자 자리 찾기, 문구 다듬기, 합치기."""

import numpy as np
from PIL import Image

from models.generative_ai.ad_layers.copywriter import clean_copy
from models.generative_ai.ad_layers.layers import (compose, find_text_boxes, load_settings, place_product,
                                                   product_canvas_mask, shadow_layer, shadow_region, to_float)

SETTINGS = load_settings()
SIZE = (256, 256)


def make_product():
    photo = Image.new("RGB", (200, 100), "white")
    mask = np.zeros((100, 200), np.float32)
    mask[20:80, 50:150] = 1
    photo.paste((200, 30, 30), (50, 20, 150, 80))
    return photo, mask


def test_product_fits_canvas_and_keeps_colors():
    photo, mask = make_product()
    product, x, y = place_product(photo, mask, SIZE, SETTINGS["product"])
    rule = SETTINGS["product"]
    assert product.width <= rule["max_share"] * SIZE[0] + 1 and product.height <= rule["max_share"] * SIZE[1] + 1
    assert abs(y + product.height - rule["bottom_y"] * SIZE[1]) <= 1
    assert product.getpixel((product.width // 2, product.height // 2)) == (200, 30, 30, 255)


def test_shadow_layer_restores_scene_brightness():
    rng = np.random.default_rng(0)
    empty = rng.uniform(0.3, 0.8, (*SIZE[::-1], 3)).astype(np.float32)
    scene = empty.copy()
    scene[180:200, 60:200] *= 0.5            # 그림자
    scene[150:170, 60:200] = np.minimum(scene[150:170, 60:200] + 0.2, 1)  # 반사
    on_product = np.zeros(SIZE[::-1], np.float32)
    on_product[100:180, 100:160] = 1
    region = shadow_region(on_product, 1.0, 0.5)
    layer = shadow_layer(scene, empty, region, on_product, 0)
    rebuilt = Image.fromarray((empty * 255).round().astype(np.uint8)).convert("RGBA")
    rebuilt.alpha_composite(layer)
    restored = to_float(rebuilt) @ np.array([0.299, 0.587, 0.114], np.float32)
    expected = scene @ np.array([0.299, 0.587, 0.114], np.float32)
    outside = on_product < 0.5
    assert np.abs(restored - expected)[outside].max() < 0.03
    assert np.asarray(layer)[..., 3][on_product > 0.5].max() == 0  # 제품 자리에는 그림자를 두지 않음


def test_text_boxes_found_outside_product_with_color():
    base = np.full((*SIZE[::-1], 3), 0.9, np.float32)
    draft = base.copy()
    for left in range(30, 220, 24):           # 제목 한 줄: 글자 8개
        draft[20:50, left:left + 16] = [0.8, 0.1, 0.1]
    draft[200:215, 30:120:6] = [0.1, 0.1, 0.1]  # 작은 부제 줄
    on_product = np.zeros(SIZE[::-1], np.float32)
    on_product[100:180, 100:160] = 1
    draft[120:140, 110:150] = 0                # 생성 모델이 제품을 조금 바꾼 것 → 무시
    boxes = find_text_boxes(draft, base, on_product, SETTINGS["text_boxes"])
    assert len(boxes) == 2
    title = boxes[0]
    assert (title["x"], title["y"], title["w"], title["h"]) == (30, 20, 184, 30)
    assert title["color"] == "#cc1a1a"
    assert boxes[1]["y"] == 200


def test_clean_copy_keeps_known_roles_once_and_uses_real_price():
    raw = {"items": [{"box": 1, "role": "title", "text": "갓 구운 케이크"},
                     {"box": 2, "role": "price", "text": "9,900원 할인"},
                     {"box": 3, "role": "title", "text": "두 번째 제목"},
                     {"box": 9, "role": "cta", "text": "없는 상자"},
                     {"box": 4, "role": "banner", "text": "모르는 역할"}]}
    copy = clean_copy(raw, 4, "6,500원")
    assert copy == [{"role": "title", "text": "갓 구운 케이크"}, {"role": "price", "text": "6,500원"},
                    {"role": "skip", "text": ""}, {"role": "skip", "text": ""}]


def test_compose_moves_shadow_with_product(tmp_path):
    Image.new("RGB", SIZE, "white").save(tmp_path / "background.png")
    Image.new("RGBA", (20, 20), (255, 0, 0, 255)).save(tmp_path / "product.png")
    Image.new("RGBA", (20, 4), (0, 0, 0, 255)).save(tmp_path / "shadow.png")
    layout = {"size": list(SIZE), "background": {"file": "background.png"},
              "product": {"file": "product.png", "x": 50, "y": 50},
              "shadow": {"file": "shadow.png", "dx": 0, "dy": 20}, "texts": []}
    assert compose(layout, tmp_path).getpixel((55, 72)) == (0, 0, 0)
    layout["product"] |= {"x": 245, "y": -10}   # 캔버스 밖으로 걸쳐도 깨지지 않음
    moved = compose(layout, tmp_path)
    assert moved.getpixel((250, 0)) == (255, 0, 0) and moved.getpixel((250, 12)) == (0, 0, 0)
    assert moved.getpixel((55, 72)) == (255, 255, 255)


def test_product_canvas_mask_matches_placement():
    photo, mask = make_product()
    product, x, y = place_product(photo, mask, SIZE, SETTINGS["product"])
    on_product = product_canvas_mask(product, x, y, SIZE)
    assert on_product[y + product.height // 2, x + product.width // 2] == 1
    assert on_product.sum() == np.asarray(product.getchannel("A"), np.float32).sum() / 255
