import json

import numpy as np
from PIL import Image
import pytest

from models.generative_ai.ad_layers import scene_ad
from models.generative_ai.photo_retouch import product_mask


@pytest.mark.parametrize("winner", [1, 2])
def test_scene_ad_plans_canvas_and_exact_copy_before_generation(tmp_path, monkeypatch, winner):
    calls = []
    photo = Image.new("RGB", (70, 60), "orange")
    reference = Image.new("RGB", (70, 60), "white")
    fields = {"product_name": "수분 세럼", "price_krw": 32000}

    def mask(model, pixels):
        assert model == "mask" and pixels[0, 0, 2] == 1
        foreground = np.zeros(pixels.shape[:2], np.float32)
        foreground[20:75, 15:47] = 1
        return foreground

    class Model:
        def use_lora(self, weights, strength):
            assert weights is None and strength == 0

        def edit_with_reference(self, original, canvas, prompt, seed, steps):
            assert original is photo and canvas.size == (512, 1024)
            assert "Amber reflections" in prompt and "부드러운 아침" not in prompt
            calls.append("qwen")
            return Image.new("RGB", canvas.size, "blue")

    class Interpreter:
        def ask(self, step, images, prompt, *, model):
            assert model == "gpt-6-luna"
            calls.append(step)
            if step == "ad_plan":
                return '{"message":"아침에 쓰는 세럼", "evidence":"입력 상품 정보"}'
            if step == "ad_direction":
                return '{"scene":"Soft window light"}'
            if step == "ad_headline":
                return '{"headline":"부드러운 아침"}'
            if step == "ad_layout_review":
                assert len(images) == 2 and images[0].shape == (1024, 1024, 3)
                return '{"reason":"Type and photo are aligned", "revision":null}'
            if step == "ad_final_layout":
                assert len(images) == 3 and images[1][0, 0, 2] == 1
                return '{"reason":"The color accent competes with the subject", "revision":{"accent":null}}'
            if step == "ad_final_compare":
                assert len(images) == 3
                return json.dumps({"winner": winner, "reason": "Clearer product focus"})
            assert step == "ad_layout" and images[0][0, 0, 2] == 0 and len(images) == 2
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            assert data["copy"] == {"headline": "부드러운 아침", "product_name": "수분 세럼", "price": "32,000원"}
            boxes = {"headline": [55, 10, 40, 30], "product_name": [55, 45, 40, 15], "price": [55, 65, 40, 15]}
            if "layout_error" not in data:
                boxes["headline"] = [55, 15, 40, 40]
            else:
                assert "겹쳤습니다" in data["layout_error"]
            style = {"font": "noto_sans_kr", "align": "left", "color": "#ffffff", "size": 5, "max_lines": 1}
            return json.dumps({"concept": "Light joins image and type", "focus": "headline", "background": "#251d16",
                               "accent": {"shape": "ellipse", "box": [35, 10, 55, 55], "color": "#9e6c4b"},
                               "artwork_box": [0, 0, 50, 100], "crop": "preserve", "edge": "framed",
                               "art_direction": "Amber reflections",
                               "boxes": boxes, "styles": {role: style for role in boxes}})

    def render(texts, size, folder):
        calls.append("render")
        assert size == (1024, 1024) and [text["x"] for text in texts] == [563, 563, 563]
        for text in texts:
            Image.new("RGBA", (text["w"], text["h"]), "white").save(folder / text["file"])

    monkeypatch.setattr(product_mask, "product_mask", mask)
    monkeypatch.setattr(scene_ad, "render_texts", render)
    folder = tmp_path / "result"
    models = {"ai": Interpreter(), "generation": Model(), "mask": "mask"}
    layout = scene_ad.make_scene_ad(photo, fields, "", models, folder, references=[reference])
    assert calls == ["ad_plan", "ad_direction", "ad_headline", "ad_layout", "ad_layout", "render", "ad_layout_review",
                     "qwen", "ad_final_layout", "render", "ad_final_compare"]
    assert (layout["design"]["accent"] is None) == (winner == 2)
    assert layout["final_review_reason"] == ("Clearer product focus" if winner == 2 else None)
    assert {layer["text"] for layer in layout["texts"]} == {"부드러운 아침", "수분 세럼", "32,000원"}
    assert json.loads((folder / "layout.json").read_text(encoding="utf-8")) == layout
    with Image.open(folder / "ad.png") as ad:
        assert ad.getpixel((15, 20)) == (0, 0, 255)
        assert ad.getpixel((563, 102)) == (255, 255, 255)
    calls.clear()
    saved = scene_ad.make_scene_ad(photo, fields, "", {"ai": None}, tmp_path / "preview",
                                   preview=True, proposal=layout)
    assert calls == ["render"]
    assert saved["design"] == layout["design"] and saved["texts"] == layout["texts"]


def test_type_can_extend_over_photo_but_not_over_product(tmp_path):
    foreground = np.zeros((30, 30), dtype=bool)
    foreground[10:20, 10:20] = True
    scene = Image.new("RGB", (30, 30))
    layout = {"size": [30, 30], "artwork": {"x": 0, "y": 0},
              "texts": [{"role": "headline", "file": "text.png", "x": 0, "y": 0, "w": 30, "h": 30}]}
    ink = Image.new("RGBA", scene.size)
    ink.putpixel((5, 12), (255, 255, 255, 255))
    ink.save(tmp_path / "text.png")
    scene_ad.check_text_overlap(foreground, layout, tmp_path)
    ink.putpixel((12, 12), (255, 255, 255, 255))
    ink.save(tmp_path / "text.png")
    with pytest.raises(ValueError, match="실제 제품을 가립니다"):
        scene_ad.check_text_overlap(foreground, layout, tmp_path)


def test_blended_photo_edges_keep_product_pixels_opaque():
    photo = Image.new("RGB", (30, 30), "orange")
    foreground = np.zeros((30, 30), dtype=bool)
    foreground[0, 15] = True
    blended = scene_ad.blend_edges(photo, foreground)
    assert blended.getpixel((0, 0))[3] == 0
    assert blended.getpixel((15, 0)) == (*photo.getpixel((15, 0)), 255)


def test_blend_only_fades_borders_inside_canvas():
    photo = Image.new("RGB", (30, 30), "orange")
    blended = scene_ad.blend_edges(photo, position=(4, 0), canvas_size=(34, 30))
    assert blended.getpixel((0, 15))[3] == 0
    assert blended.getpixel((29, 0))[3] == blended.getpixel((29, 29))[3] == 255
