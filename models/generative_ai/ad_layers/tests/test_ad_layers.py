"""그래픽 카드와 모델 없이 도는 확인: 제품 놓기, 접지 그림자, 글자 배치, 문구 다듬기, 합치기."""

import numpy as np
from PIL import Image

from models.generative_ai.ad_layers.copywriter import clean_copy
from models.generative_ai.ad_layers.layers import (compose, contact_shadow, load_settings, place_product,
                                                   template_boxes)
from models.generative_ai.ad_layers import make_ad
from models.generative_ai.photo_retouch import product_mask

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


def test_contact_shadow_darkens_only_near_the_footprint():
    product = Image.new("RGBA", (100, 80), (200, 30, 30, 255))
    shadow, dx, dy = contact_shadow(product, overhead=False)
    canvas = Image.new("RGBA", SIZE, "white")
    canvas.alpha_composite(shadow, (50 + dx, 60 + dy))
    assert canvas.getpixel((100, 90)) == (255, 255, 255, 255)
    at_base = canvas.getpixel((100, 140))
    assert at_base[0] == at_base[1] == at_base[2] < 255
    assert at_base[0] < canvas.getpixel((100, 144))[0] < 255
    assert canvas.getpixel((100, 155)) == (255, 255, 255, 255)
    overhead, _, overhead_dy = contact_shadow(product, overhead=True)
    assert overhead.height > shadow.height and overhead_dy < dy


def test_contact_shadow_follows_uneven_product_bases():
    product = Image.new("RGBA", (100, 80))
    product.paste((200, 30, 30, 255), (0, 0, 55, 50))
    product.paste((200, 30, 30, 255), (50, 35, 100, 80))
    shadow, dx, dy = contact_shadow(product, overhead=False)
    canvas = Image.new("RGBA", SIZE, "white")
    canvas.alpha_composite(shadow, (50 + dx, 60 + dy))
    assert canvas.getpixel((75, 110))[0] < 255
    assert canvas.getpixel((125, 140))[0] < 255
    assert canvas.getpixel((75, 140))[0] == 255


def test_text_template_picks_readable_color():
    base = np.full((*SIZE[::-1], 3), 0.9, np.float32)
    boxes = template_boxes(base, SETTINGS["template"])
    assert len(boxes) == len(SETTINGS["template"]) and boxes[0]["color"] == "#1a1a1a"
    assert template_boxes(base * 0.1, SETTINGS["template"])[0]["color"] == "#ffffff"


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


def test_make_ad_generates_only_background_and_renders_exact_price(tmp_path, monkeypatch):
    photo, mask = make_product()
    monkeypatch.setattr(product_mask, "product_mask", lambda model, pixels: mask)

    class Model:
        def use_lora(self, file, strength):
            assert file is None and strength == 0

        def edit(self, image, prompt, seed, steps):
            assert "blank canvas" in prompt
            assert image.getextrema() == ((128, 128),) * 3
            assert "directly above" in prompt
            return Image.new("RGB", image.size, "#cfcfcf")

    class Interpreter:
        def ask(self, purpose, images, prompt):
            if purpose == "ad_scene":
                return '{"scene": "An empty stone countertop", "view": "overhead"}'
            assert "지정 역할 title" in prompt and images[0][700, 512, 0] > images[0][700, 512, 1]
            return '{"items": [{"box": 1, "role": "title", "text": "딸기 케이크"}]}'

    def render(texts, size, folder):
        for line in texts:
            Image.new("RGBA", (line["w"], line["h"])).save(folder / line["file"])

    monkeypatch.setattr(make_ad, "render_texts", render)
    fields = {"product_name": "딸기 케이크", "unit": "1조각", "price_krw": 6500}
    layout = make_ad.make_ad(photo, fields, "minimal", "", {"generation": Model(), "mask": None, "ai": Interpreter()}, tmp_path)
    assert layout["view"] == "overhead" and layout["text_source"] == "template"
    assert [(line["role"], line["text"]) for line in layout["texts"]] == [("title", "딸기 케이크"), ("price", "6,500원")]
    with Image.open(tmp_path / "background.png") as background:
        assert background.getextrema() == ((207, 207),) * 3
    assert not (tmp_path / "draft.png").exists()

    (tmp_path / "base.png").unlink()
    make_ad.relayout(tmp_path, Interpreter())
    assert (tmp_path / "base.png").is_file()
    assert (tmp_path / "ad.png").is_file()
