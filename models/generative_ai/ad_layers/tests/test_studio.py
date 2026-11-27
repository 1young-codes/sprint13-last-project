import json
import shutil

from PIL import Image
import pytest

from models.generative_ai.ad_layers import scene_ad, studio, studio_edits, studio_formats
from models.generative_ai.ad_layers.layers import compose, load_layout, save_layout


CAMPAIGN_READY = ('{"scene":{"treatment":"staging","observation":"촛불을 밝혀 저녁을 표현합니다"},'
                  '"layout":{"ready":true,"observation":"촛불과 문구가 한 축에 보입니다"}}')


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
                if data["version_order"] == ["v01"]:
                    assert images[2].shape == (768, 768, 3)
                    assert images[2][400, 400].tolist() == [0, 0, 1]
                    return '{"version":"v01","finding":"typography","observation":"제목이 어색합니다"}'
                assert data["version_order"] == ["v01", "v02"]
                assert images[1][10, 10].tolist() == [1, 1, 1]
                assert images[2][10, 10].tolist() == [0, 0, 1]
                assert images[3][400, 150].tolist() == images[3][400, 540].tolist() == [0, 0, 1]
                return '{"version":"v02","finding":"ready","observation":"제목을 다시 배치했습니다"}'
            if step == "studio_campaign_check":
                assert len(images) == 4
                return CAMPAIGN_READY
            if step == "studio_reframe":
                assert data["size"] == [600, 600]
                assert data["approved_design"]["boxes"]["headline"] == [20, 20, 45, 10]
                assert data["preferred_first_lines"] == {"headline": "포근한 겨울"}
                assert data["min_box_height_percent"] == {"one_line": 3, "two_lines": 5}
                updated = data["approved_design"]
                updated["styles"]["headline"]["break_after"] = 1
                return json.dumps(updated)
            if "edit_text" in data["actions"]:
                assert set(data["actions"]) == {"edit_text", "hold"}
                return json.dumps({"action": "edit_text", "arguments": {"changes": {"headline": {
                    "box": [20, 20, 45, 10], "first_line": "포근한 겨울",
                    "emphasis": {"phrase": "포근한 겨울", "color": "#FFAA33"}}}}, "reason": "제목 이동"})
            assert set(data["actions"]) == {"select", "hold"}
            edit = data["earlier_edits"][0]
            assert (edit["action"], edit["candidate"], edit["preferred_version"], edit["remaining_finding"]) == (
                "edit_text", "v02", "v02", "ready")
            return '{"action":"select","arguments":{},"reason":"제목과 제품이 함께 읽힙니다"}'

    output = tmp_path / "output"
    result = studio.run_studio(Image.new("RGB", (100, 100)), fields, "", output, Agent(), draft=folder,
                               formats=("thumbnail",))
    assert stages == ["studio_look", "studio_action", "studio_look", "studio_campaign_check", "studio_action", "studio_reframe"]
    assert result["status"] == "complete" and result["version"] == "v02"
    assert (output / "final" / "scene.png").read_bytes() == (folder / "scene.png").read_bytes()
    assert (output / "final" / "text_price.png").read_bytes() == (folder / "text_price.png").read_bytes()
    assert next(layer["text"] for layer in load_layout(output / "final")["texts"] if layer["role"] == "price") == "18,000원"
    assert load_layout(output / "final")["design"]["styles"]["headline"]["break_after"] == 2
    assert load_layout(output / "formats/thumbnail")["design"]["styles"]["headline"]["break_after"] == 2
    assert load_layout(output / "formats/thumbnail")["design"]["styles"]["headline"]["emphasis"] == {
        "phrase": "포근한 겨울", "color": "#FFAA33"}
    assert (output / "versions/v01/ad.png").read_bytes() == (folder / "ad.png").read_bytes()
    with Image.open(result["formats"]["thumbnail"]) as poster:
        assert poster.size == (600, 600)


def test_composition_edit_softens_the_edge_without_repainting(tmp_path, draft):
    folder, fields = draft
    layout = load_layout(folder)
    layout["artwork"].update(x=30, w=70)
    layout["design"]["artwork_box"] = [30, 0, 70, 100]
    mask = Image.new("L", (70, 100))
    mask.paste(255, (35, 30, 55, 80))
    mask.save(folder / "foreground.png")
    scene = Image.new("RGBA", mask.size, "blue")
    scene.putpixel((40, 40), (0, 0, 255, 170))
    scene.save(folder / "scene.png")
    compose(layout, folder).save(folder / "ad.png")
    save_layout(layout, folder)

    class Agent:
        def ask(self, step, images, request, *, model):
            data = json.loads(request.split("Input JSON follows:\n", 1)[1])
            if step == "studio_look":
                if data["version_order"] == ["v01"]:
                    return '{"version":"v01","finding":"layout","observation":"사진 경계가 글자와 제품을 갈라놓습니다"}'
                return '{"version":"v02","finding":"ready","observation":"제품과 제목이 이어집니다"}'
            if step == "studio_campaign_check":
                return CAMPAIGN_READY
            if step == "ad_final_layout":
                assert data["composition_issue"] == "제목과 제품 사이의 세로 경계를 부드럽게 이어 주세요"
                if data["layout_error"] is None:
                    return '{"reason":"기존 배치를 유지합니다","revision":null}'
                assert "배치 수정이 빠졌습니다" in data["layout_error"]
                boxes = data["design"]["boxes"] | {"headline": [27, 10, 34, 20]}
                return json.dumps({"reason": "제품 쪽에 제목을 붙였습니다", "revision": {
                    "edge": "blend", "boxes": boxes, "accent": {"phrase": "포근한 겨울", "color": "#FFAA33"}}})
            if "edit_composition" in data["actions"]:
                assert data["actual_foreground_bounds"] == [65, 30, 85, 80]
                assert data["rendered_text_bounds"]["headline"] == [10, 10, 13, 13]
                return json.dumps({"action": "edit_composition", "reason": "제목을 사진 쪽으로 옮깁니다",
                                   "arguments": {"instruction": "제목과 제품 사이의 세로 경계를 부드럽게 이어 주세요"}})
            return '{"action":"select","arguments":{},"reason":"경계가 사라졌습니다"}'

    output = tmp_path / "composition-edit"
    result = studio.run_studio(Image.new("RGB", (100, 100)), fields, "", output, Agent(), draft=folder, edit_limit=1)
    assert result["status"] == "complete"
    revised = load_layout(output / "final")
    assert revised["artwork"] == layout["artwork"] and revised["design"]["edge"] == "blend"
    assert revised["design"]["boxes"]["headline"] == [27, 10, 34, 20]
    assert revised["design"]["styles"]["headline"]["emphasis"] == {"phrase": "포근한 겨울", "color": "#FFAA33"}
    assert revised["design"]["accent"] is None
    assert [row["text"] for row in revised["texts"]] == [row["text"] for row in layout["texts"]]
    with Image.open(output / "final/scene.png") as image:
        assert image.getpixel((0, 40)) == (0, 0, 255, 0)
        assert image.getpixel((40, 40)) == scene.getpixel((40, 40))
        assert image.convert("RGB").tobytes() == scene.convert("RGB").tobytes()


@pytest.mark.parametrize(("finding", "observation", "has_reference"), [
    ("scene", "원본 촛불과 다른 연출이 보이지 않습니다", False),
    ("scene", "원본 촛불과 다른 연출이 보이지 않습니다", True),
    ("layout", "제목과 가격이 서로 어긋나 별개로 읽힙니다", True),
])
def test_campaign_comparison_can_block_selection(tmp_path, draft, finding, observation, has_reference):
    folder, fields = draft
    if finding == "scene":
        Image.new("RGB", (100, 100), "green").save(folder / "composition.png")
    references = [Image.new("RGB", (100, 100), "red")] if has_reference else []

    class Agent:
        def ask(self, step, images, request, *, model):
            if step == "studio_look":
                return '{"version":"v01","finding":"ready","observation":"중앙 정렬이 깔끔합니다"}'
            if step == "studio_campaign_check":
                has_study = finding == "scene"
                assert len(images) == 4 + has_study + has_reference and images[1][10, 10].tolist() == [1, 1, 1]
                assert images[2][10, 10].tolist() == [0, 0, 1]
                assert images[3].shape == (768, 768, 3)
                if has_study:
                    assert images[4][0, 0, 1] > 0 and images[4][0, 0, 0] == 0
                if has_reference:
                    assert images[-1][0, 0].tolist() == [1, 0, 0]
                data = json.loads(request.split("Input JSON follows:\n", 1)[1])
                assert data["has_source_study"] == has_study
                assert data["rendered_text_bounds"]["headline"] == [10, 10, 13, 13]
                assert "포근한 겨울 저녁" in request
                return json.dumps({
                    "scene": {"treatment": "catalog" if finding == "scene" else "lighting",
                              "observation": observation if finding == "scene" else "촛불이 잘 보입니다"},
                    "layout": {"ready": finding != "layout", "observation": observation if finding == "layout" else "문구가 연결됩니다"},
                    "reference_observation": "참고작의 그림자가 불빛의 위치를 드러냅니다" if has_reference else None,
                })
            assert step == "studio_action"
            assert len(images) == 2 and images[0][10, 10].tolist() == [1, 1, 1]
            assert images[1][10, 10].tolist() == [0, 0, 0]
            data = json.loads(request.split("Input JSON follows:\n", 1)[1])
            assert list(data["actions"]) == ["hold"]
            assert data["finding"] == finding and data["observation"] == observation and data["owner_idea"] == ""
            return '{"action":"hold","arguments":{},"reason":"촬영부터 고쳐야 합니다"}'

    output = tmp_path / "output"
    result = studio.run_studio(Image.new("RGB", (100, 100)), fields, "", output, Agent(), draft=folder,
                               references=references, edit_limit=0)
    assert result["status"] == "needs_review" and not (output / "final").exists()
    held = result["image"].parent
    assert load_layout(held)["pending_review"]["finding"] == finding

    class Resume(Agent):
        def ask(self, step, images, request, *, model):
            assert step == "studio_action"
            assert observation in request
            return super().ask(step, images, request, model=model)

    resumed = studio.run_studio(Image.new("RGB", (100, 100)), fields, "", tmp_path / "resumed", Resume(),
                                draft=held, edit_limit=0)
    assert resumed["status"] == "needs_review" and not (tmp_path / "resumed/final").exists()

    with Image.open(held / "ad.png") as poster:
        changed = poster.copy()
    changed.putpixel((0, 0), (255, 0, 0))
    changed.save(held / "ad.png")
    fresh = tmp_path / "fresh-review"
    studio.run_studio(Image.new("RGB", (100, 100)), fields, "", fresh, Agent(), draft=held,
                      references=references, edit_limit=0)
    assert "resume_review" not in (fresh / "events.jsonl").read_text(encoding="utf-8")


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
    cropped, visible = studio_formats.frame_photo(
        photo, mask, (50, 100), lambda *_: pytest.fail("자를 수 있으면 배경을 생성할 필요가 없습니다"))
    assert cropped.getpixel((0, 0)) == (0, 0, 255)
    assert visible.getbbox() == (29, 25, 50, 76)

    mask.paste(255, (7, 25, 93, 76))
    padded, visible = studio_formats.frame_photo(photo, mask, (50, 100))
    assert padded.getpixel((0, 0)) == (0, 0, 255)
    assert visible.getbbox() == (2, 36, 48, 63)

    def expand(frame, margin):
        assert frame.size == (50, 100)
        assert margin[:23].all() and margin[76:].all() and not margin[23:76].any()
        return Image.new("RGB", frame.size, "green")

    expanded, protected = studio_formats.frame_photo(photo, mask, (50, 100), expand)
    assert expanded.getpixel((0, 0)) == (0, 128, 0)
    assert expanded.getpixel((25, 50)) == (0, 0, 255)
    assert protected.tobytes() == visible.tobytes()
    known = (0, 23, 50, 76)
    assert expanded.crop(known).tobytes() == padded.crop(known).tobytes()


def test_format_delegates_new_breaks_to_copy_review():
    styles = {"headline": {"max_lines": 2, "break_after": 1}, "product_name": {"max_lines": 2}}
    layers = [{"role": "headline", "text": "따뜻한 빛으로 저녁을 감싸요"},
              {"role": "product_name", "text": "세라믹 향초"}]

    class Agent:
        def ask(self, step, images, request, *, model):
            assert step == "studio_copy_break" and not images
            return '{"headline":"따뜻한 빛으로","product_name":null}'

    first_lines = {}
    updated = studio_formats.format_styles(Agent(), styles, layers, first_lines)
    assert updated == {"headline": {"max_lines": 2, "break_after": 2}, "product_name": {"max_lines": 1}}
    assert studio_formats.format_styles(None, styles, layers, first_lines) == updated


def test_expands_only_the_chosen_photo_placement():
    photo = Image.new("RGB", (100, 100), "blue")
    mask = Image.new("L", photo.size)
    mask.paste(255, (7, 25, 93, 76))
    expanded = []

    def choose(previews):
        assert list(previews) == ["top", "center", "bottom"]
        assert all(preview.size == (50, 100) for preview in previews.values())
        return "bottom"

    def fill(frame, margin):
        assert margin[:39].all() and margin[92:].all() and not margin[39:92].any()
        expanded.append(frame)
        return Image.new("RGB", frame.size, "green")

    result, visible = studio_formats.frame_photo(photo, mask, (50, 100), fill, choose)
    assert len(expanded) == 1 and visible.getbbox() == (2, 52, 48, 79)
    assert result.crop((0, 39, 50, 92)).tobytes() == expanded[0].crop((0, 39, 50, 92)).tobytes()


def test_composition_review_can_fix_a_valid_layout(draft):
    folder, _ = draft
    layout = load_layout(folder)
    design = layout["design"]
    foreground = studio_edits.foreground_mask(folder, layout)
    copy = {layer["role"]: layer["text"] for layer in layout["texts"]}
    original_photo = (folder / "scene.png").read_bytes()

    class Agent:
        def ask(self, step, images, request, *, model):
            assert step == "ad_final_layout" and len(images) == 2
            boxes = design["boxes"] | {"product_name": [10, 37, 45, 7], "price": [10, 45, 45, 7]}
            styles = design["styles"] | {"headline": design["styles"]["headline"] | {
                "emphasis": {"phrase": "겨", "color": "#ff0000"}}}
            return json.dumps({"reason": "판매 정보를 묶었습니다", "revision": {"boxes": boxes, "styles": styles}})

    with Image.open(folder / "scene.png") as source:
        revised, _, why = scene_ad.review_finished(Agent(), design, layout, source.copy(), foreground, folder,
                                                   {"copy": copy, "fonts": ["noto_sans_kr"]}, [],
                                                   assess_composition=True)
    assert why == "판매 정보를 묶었습니다" and revised["boxes"]["price"] == [10, 45, 45, 7]
    assert revised["styles"] == design["styles"]
    assert (folder / "scene.png").read_bytes() == original_photo


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
            if step == "studio_campaign_check":
                return CAMPAIGN_READY
            return '{"action":"select","arguments":{},"reason":"촛농과 제품을 잘 보여 줍니다"}'

    ai = Agent()

    def make(photo, facts, idea, models, output, *, references, start_direction):
        assert facts == fields and idea == "재질 강조"
        assert "Show the candle wax" in start_direction
        assert not references and models["ai"] is ai
        shutil.copytree(folder, output)

    monkeypatch.setattr(studio, "make_scene_ad", make)
    result = studio.run_studio(Image.new("RGB", (100, 100)), fields, "재질 강조", tmp_path / "output", ai,
                               lambda: {"generation": object()})
    assert result["status"] == "complete" and result["image"].is_file()
