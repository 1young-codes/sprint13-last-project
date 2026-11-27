import json

import numpy as np
from PIL import Image

from models.generative_ai.ad_layers import scene_ad
from models.generative_ai.photo_retouch import product_mask


def test_scene_ad_places_exact_copy_on_the_generated_scene(tmp_path, monkeypatch):
    calls = []
    photo = Image.new("RGB", (70, 60), "orange")
    scene = Image.new("RGB", (100, 100), "blue")
    foreground = np.zeros((100, 100), np.float32)
    foreground[20:75, 15:47] = 1
    fields = {"product_name": "수분 세럼", "price_krw": 32000}

    def mask(model, pixels):
        assert model == "mask" and pixels[0, 0, 2] == 1
        return foreground

    class Model:
        def use_lora(self, weights, strength):
            assert weights is None and strength == 0

        def edit_with_reference(self, original, canvas, prompt, seed, steps):
            assert original is photo and canvas.width == canvas.height
            assert "quiet area at the right" in prompt and "부드러운 아침" not in prompt
            calls.append("qwen")
            return scene

    class Interpreter:
        def ask(self, step, images, prompt, *, model):
            assert model == "gpt-6-luna"
            calls.append(step)
            if step == "ad_plan":
                return '{"message":"아침에 쓰는 세럼", "evidence":"입력 상품 정보"}'
            if step == "ad_direction":
                return '{"scene":"Soft window light", "focus":"product", "copy_space":"right"}'
            if step == "ad_headline":
                return '{"headline":"부드러운 아침"}'
            assert step == "ad_layout" and images[0][0, 0, 2] == 1
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            assert data["copy"] == {"headline": "부드러운 아침", "product_name": "수분 세럼", "price": "32,000원"}
            assert data["product_bounds_pixels"] == [15, 20, 47, 75]
            boxes = {"headline": [55, 10, 40, 30], "product_name": [55, 45, 40, 15], "price": [55, 65, 40, 15]}
            if "layout_error" not in data:
                boxes["headline"] = [15, 15, 40, 25]
            else:
                assert "실제 제품" in data["layout_error"]
            style = {"font": "noto_sans_kr", "align": "left", "color": "#ffffff", "size": 5, "max_lines": 1}
            return json.dumps({"boxes": boxes, "styles": {role: style for role in boxes}})

    def render(texts, size, folder):
        assert size == scene.size and [text["x"] for text in texts] == [55, 55, 55]
        for text in texts:
            Image.new("RGBA", (text["w"], text["h"]), "white").save(folder / text["file"])

    monkeypatch.setattr(product_mask, "product_mask", mask)
    monkeypatch.setattr(scene_ad, "render_texts", render)
    folder = tmp_path / "result"
    layout = scene_ad.make_scene_ad(photo, fields, "", {"ai": Interpreter(), "generation": Model(), "mask": "mask"}, folder)
    assert calls == ["ad_plan", "ad_direction", "ad_headline", "qwen", "ad_layout", "ad_layout"]
    assert {layer["text"] for layer in layout["texts"]} == {"부드러운 아침", "수분 세럼", "32,000원"}
    assert json.loads((folder / "layout.json").read_text(encoding="utf-8")) == layout
    with Image.open(folder / "ad.png") as ad:
        assert ad.getpixel((15, 20)) == (0, 0, 255)
        assert ad.getpixel((55, 10)) == (255, 255, 255)
