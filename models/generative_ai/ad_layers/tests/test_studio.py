import json
import shutil

from PIL import Image
import pytest

from models.generative_ai.ad_layers import scene_ad, studio, studio_edits, studio_formats
from models.generative_ai.ad_layers.layers import compose, load_layout, save_layout


@pytest.fixture
def draft(tmp_path, monkeypatch):
    def render(texts, size, folder):
        for text in texts:
            layer = Image.new("RGBA", (text["w"], text["h"]))
            layer.paste(text["color"], (0, 0, 3, 3))
            layer.save(folder / text["file"])

    monkeypatch.setattr(scene_ad, "render_texts", render)
    monkeypatch.setattr(studio_edits, "render_texts", render)
    folder = tmp_path / "draft"
    folder.mkdir()
    fields = {"product_name": "세라믹 향초", "price_krw": 18000}
    copy = {"headline": "포근한 겨울 저녁", "product_name": fields["product_name"], "price": "18,000원"}
    style = {"font": "noto_sans_kr", "align": "left", "color": "#ffffff", "size": 5, "max_lines": 1}
    design = {"concept": "Warm indoor light", "art_direction": "A candle on a wooden table", "crop": "preserve",
              "edge": "framed", "focus": "product", "background": "#222222", "accent": None,
              "artwork_box": [0, 0, 100, 100],
              "boxes": {"headline": [10, 10, 45, 20], "product_name": [10, 35, 45, 10], "price": [10, 50, 45, 10]},
              "styles": dict.fromkeys(copy, style)}
    layout = scene_ad.design_layers(design, copy, (100, 100), ["noto_sans_kr"], folder)
    layout.update(design=design, fields=fields, brief={"headline": copy["headline"]}, scene_prompt=design["art_direction"])
    mask = Image.new("L", (100, 100))
    mask.paste(255, (65, 30, 92, 88))
    mask.save(folder / "foreground.png")
    Image.new("RGB", (100, 100), "blue").save(folder / "scene.png")
    compose(layout, folder).save(folder / "ad.png")
    save_layout(layout, folder)
    return folder, fields


def test_observe_search_edit_and_select_the_best_version(tmp_path, draft):
    folder, fields = draft
    stages = []

    class Agent:
        def ask(self, step, images, request, *, model):
            assert model == "gpt-6-luna"
            stages.append(step)
            data = json.loads(request.split("Input JSON follows:\n", 1)[1])
            if step == "studio_look":
                if len(images) == 2:
                    return '{"version":"v01","finding":"typography","observation":"제목이 어색합니다"}'
                assert data["version_order"] == ["v01", "v02"]
                assert images[1][10, 10].tolist() == [1, 1, 1]
                assert images[2][10, 10].tolist() == [0, 0, 1]
                return '{"version":"v02","finding":"ready","observation":"제목을 다시 배치했습니다"}'
            if step == "studio_reframe":
                assert data["size"] == [600, 600]
                assert data["approved_design"]["boxes"]["headline"] == [20, 20, 45, 10]
                return json.dumps(data["approved_design"])
            if "edit_text" in data["actions"]:
                assert set(data["actions"]) == {"edit_text", "hold"}
                return json.dumps({"action": "edit_text", "arguments": {"changes": {"headline": {
                    "box": [20, 20, 45, 10], "first_line": "포근한 겨울"}}}, "reason": "제목 이동"})
            assert set(data["actions"]) == {"select", "hold"}
            return '{"action":"select","arguments":{},"reason":"제목과 제품이 함께 읽힙니다"}'

    output = tmp_path / "output"
    result = studio.run_studio(Image.new("RGB", (100, 100)), fields, "", output, Agent(), draft=folder,
                               formats=("thumbnail",))
    assert stages == ["studio_look", "studio_action", "studio_look", "studio_action", "studio_reframe"]
    assert result["status"] == "complete" and result["version"] == "v02"
    assert (output / "final" / "scene.png").read_bytes() == (folder / "scene.png").read_bytes()
    assert (output / "final" / "text_price.png").read_bytes() == (folder / "text_price.png").read_bytes()
    assert next(layer["text"] for layer in load_layout(output / "final")["texts"] if layer["role"] == "price") == "18,000원"
    assert load_layout(output / "final")["design"]["styles"]["headline"]["break_after"] == 2
    assert (output / "versions/v01/ad.png").read_bytes() == (folder / "ad.png").read_bytes()
    with Image.open(result["formats"]["thumbnail"]) as poster:
        assert poster.size == (600, 600)


def test_region_edit_locks_product_and_rest_of_photo(tmp_path, draft):
    folder, _ = draft

    class Generator:
        def use_lora(self, *_):
            pass

        def edit(self, _image, prompt, seed, steps, *, size):
            assert "Remove the prop" in prompt
            return Image.new("RGB", size, "green")

    model = {"generation": Generator()}
    edited = tmp_path / "edited"
    studio_edits.edit_version(folder, edited, "edit_region", {"box": [25, 65, 30, 25], "instruction": "Remove the prop"},
                              lambda: model)
    result = Image.open(edited / "scene.png")
    source = Image.open(folder / "scene.png").convert("RGBA")
    assert result.getpixel((40, 77)) == (0, 128, 0, 255)
    assert result.crop((55, 0, 100, 100)).tobytes() == source.crop((55, 0, 100, 100)).tobytes()
    blocked = tmp_path / "blocked"
    with pytest.raises(ValueError, match="제품에 닿습니다"):
        studio_edits.edit_version(folder, blocked, "edit_region",
                                  {"box": [60, 35, 20, 20], "instruction": "Remove the prop"}, lambda: model)
    assert not blocked.exists()


def test_crops_around_subject_and_pads_when_crop_would_cut_it():
    photo = Image.new("RGB", (100, 100), "blue")
    mask = Image.new("L", photo.size)
    mask.paste(255, (72, 25, 93, 76))
    cropped, visible = studio_formats.frame_photo(photo, mask, (50, 100))
    assert cropped.getpixel((0, 0)) == (0, 0, 255)
    assert visible.getbbox() == (29, 25, 50, 76)

    mask.paste(255, (7, 25, 93, 76))
    padded, visible = studio_formats.frame_photo(photo, mask, (50, 100))
    assert padded.getpixel((0, 0)) == (0, 0, 255)
    assert visible.getbbox() == (3, 37, 46, 63)


def test_import_rejects_a_draft_with_modified_price(tmp_path, draft):
    folder, fields = draft
    layout = load_layout(folder)
    next(layer for layer in layout["texts"] if layer["role"] == "price")["text"] = "1,000원"
    save_layout(layout, folder)
    with pytest.raises(ValueError, match="가격 또는 확정 문구"):
        studio.run_studio(Image.new("RGB", (100, 100)), fields, "", tmp_path / "output", None, draft=folder)


def test_studio_creates_the_first_draft_with_the_selected_route(tmp_path, draft, monkeypatch):
    folder, fields = draft

    class Agent:
        def ask(self, step, images, request, *, model):
            if step == "studio_start":
                assert "reference_poster" not in request and len(images) == 1
                return '{"route":"product_shoot","instruction":"Show the candle wax in warm light","reason":"재질 강조"}'
            if step == "studio_look":
                return '{"version":"v01","finding":"ready","observation":"제품의 촛농 질감이 자연스럽습니다"}'
            return '{"action":"select","arguments":{},"reason":"촛농과 제품을 잘 보여 줍니다"}'

    ai = Agent()

    def make(photo, facts, idea, models, output, *, references):
        assert facts == fields and "Show the candle wax" in idea
        assert not references and models["ai"] is ai
        shutil.copytree(folder, output)

    monkeypatch.setattr(studio, "make_scene_ad", make)
    result = studio.run_studio(Image.new("RGB", (100, 100)), fields, "재질 강조", tmp_path / "output", ai,
                               lambda: {"generation": object()})
    assert result["status"] == "complete" and result["image"].is_file()
