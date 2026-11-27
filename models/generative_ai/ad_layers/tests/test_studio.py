import json
import shutil

from PIL import Image
import pytest
import numpy as np

from models.generative_ai.ad_layers import format_layout, scene_ad, studio, studio_edits, studio_formats
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


@pytest.mark.parametrize("local_offer", [False, True])
def test_observe_search_edit_and_select_the_best_version(tmp_path, draft, monkeypatch, local_offer):
    monkeypatch.setattr(studio_formats, "position_options", lambda *_: {})
    monkeypatch.setattr(studio_formats, "format_measurements", lambda *_: {})
    folder, fields = draft
    stages = []

    class Agent:
        def ask(self, step, images, request, *, model):
            assert model == "gpt-6-luna"
            stages.append(step)
            data = json.loads(request.split("Input JSON follows:\n", 1)[1])
            if step == "studio_look":
                if data["version_order"] == ["v01"]:
                    assert images[2].shape == (360, 360, 3)
                    assert images[2][180, 180].tolist() == [0, 0, 1]
                    return '{"version":"v01","finding":"typography","observation":"제목이 어색합니다"}'
                assert data["version_order"] == ["v01", "v02"]
                assert images[1][10, 10].tolist() == [1, 1, 1]
                assert images[2][10, 10].tolist() == [0, 0, 1]
                assert images[3].shape == (360, 732, 3)
                assert images[3][180, 180].tolist() == images[3][180, 552].tolist() == [0, 0, 1]
                return '{"version":"v02","finding":"ready","observation":"제목을 다시 배치했습니다"}'
            if step == "studio_campaign_check":
                assert len(images) == 4
                return CAMPAIGN_READY
            if step == "studio_reframe":
                assert data["size"] == [600, 600]
                assert data["approved_design"]["boxes"]["headline"] == [20, 20, 45, 10]
                assert data["preferred_first_lines"] == {"headline": "포근한 겨울"}
                assert data["min_box_height_percent"] == {"one_line": 3, "two_lines": 5}
                assert len(images) == (4 if "layout_error" in data else 3)
                updated = data["approved_design"]
                updated["styles"]["headline"]["break_after"] = 1
                if "layout_error" in data:
                    assert "제목이 사진 속 상품 윤곽에 붙어 보입니다" in data["layout_error"]
                    assert not local_offer
                    updated["boxes"]["price"] = [10, 47, 45, 10]
                return json.dumps(updated)
            if step == "studio_format_offer":
                assert local_offer and data["observation"] == "판매 정보가 떨어져 보입니다"
                assert len(images) == 1 + len(data["position_order"])
                assert all(np.any(images[0] != picture) for picture in images[1:])
                return json.dumps({"position": data["position_order"][0], "observation": "글자를 함께 옮깁니다"})
            if step == "studio_format_reading":
                assert len(images) == 6 and data["copy"]["price"] == "18,000원"
                assert all(crop < poster for crop, poster in zip(images[3].shape[:2], images[2].shape[:2]))
                assert all(crop < poster for crop, poster in zip(images[4].shape[:2], images[2].shape[:2]))
                assert images[4].shape == images[5].shape and np.any(images[4] != images[5])
                assert data["headline_behind_subject"] is False
                if stages.count(step) <= (2 if local_offer else 1):
                    return json.dumps({"finding": "offer" if local_offer else "ready",
                                       "headline_touches_product": not local_offer,
                                       "observation": "판매 정보가 떨어져 보입니다"})
                return '{"finding":"ready","headline_touches_product":false,"observation":"판매 정보가 제목과 연결됩니다"}'
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
    assert stages == ["studio_look", "studio_action", "studio_look", "studio_campaign_check", "studio_action",
                      "studio_reframe", "studio_format_reading"] + (
                         ["studio_format_offer", "studio_format_reading"] * 2 if local_offer
                         else ["studio_reframe", "studio_format_reading"])
    assert result["status"] == "complete" and result["version"] == "v02"
    assert (output / "final" / "scene.png").read_bytes() == (folder / "scene.png").read_bytes()
    assert (output / "final" / "text_price.png").read_bytes() == (folder / "text_price.png").read_bytes()
    assert next(layer["text"] for layer in load_layout(output / "final")["texts"] if layer["role"] == "price") == "18,000원"
    assert load_layout(output / "final")["design"]["styles"]["headline"]["break_after"] == 2
    assert load_layout(output / "formats/thumbnail")["design"]["styles"]["headline"]["break_after"] == 2
    assert load_layout(output / "formats/thumbnail")["design"]["styles"]["headline"]["emphasis"] == {
        "phrase": "포근한 겨울", "color": "#FFAA33"}
    boxes = load_layout(output / "formats/thumbnail")["design"]["boxes"]
    assert boxes["headline"] == [20, 20, 45, 10]
    if local_offer:
        assert boxes["product_name"] != [10, 35, 45, 10]
        assert boxes["price"][0] == boxes["product_name"][0]
        assert boxes["price"][1] - boxes["product_name"][1] == 15
        assert boxes["price"][2:] == boxes["product_name"][2:] == [45, 10]
    else:
        assert boxes["product_name"] == [10, 35, 45, 10]
        assert boxes["price"] == [10, 47, 45, 10]
    assert (output / "versions/v01/ad.png").read_bytes() == (folder / "ad.png").read_bytes()
    with Image.open(result["formats"]["thumbnail"]) as poster:
        assert poster.size == (600, 600)


@pytest.mark.parametrize(("staged", "ready", "same_piece"), [(True, True, True), (False, False, True), (True, True, False)])
def test_native_format_reuses_approved_copy_and_publishes_only_after_review(
        tmp_path, draft, monkeypatch, staged, ready, same_piece):
    folder, fields = draft
    approved = load_layout(folder)
    approved["photo_cue"] = "The vessel reflects a strip of lamp light."
    approved["brief"] = {"headline": "다듬기 전 제목", "message": "포근한 겨울 저녁을 보여 줍니다"}
    approved["source_framing"] = {"source_crop": None, "reason": "용기까지 보여 줍니다"}
    approved["art_direction_plan"] = {"theme": "lifestyle", "angle": "three_quarter", "shot_scale": "whole",
                                      "lighting": "motivated", "evidence": "실제 용기", "photo_device": "Warm lamp light"}
    save_layout(approved, folder)
    photo = Image.new("RGB", (100, 100), "orange")
    calls = []
    issue = studio_formats.FormatPhotoIssue("세로 축에 사진의 단서가 없습니다", Image.new("RGB", (1080, 1920)))

    class Model:
        def use_lora(self, *_):
            pass

        def edit(self, reference, prompt, seed, steps, *, size):
            assert reference.size == size == (768, 1365)
            assert prompt.startswith("Compose for this vertical photograph")
            assert "Warm lamp light" in prompt
            calls.append("photo")
            return Image.new("RGB", size, "orange")

        def edit_with_reference(self, original, guide, prompt, seed, steps):
            assert original.size == (640, 640) and original.getpixel((0, 0)) == photo.getpixel((0, 0))
            assert "Keep each piece once" in prompt
            return self.edit(guide, prompt, seed, steps, size=guide.size)

    class Agent:
        def ask(self, step, images, request, *, model):
            calls.append(step)
            data = json.loads(request.split("Input JSON follows:\n", 1)[1])
            if step == "ad_format_staging":
                assert data["target_canvas"] == [1080, 1920] and len(images) == 3
                return '{"move":"분리된 작은 용기를 같은 탁자의 앞쪽으로 옮깁니다"}'
            if step == "ad_staging":
                return '{"source_point":[25,40],"target_base":[40,90],"reason":"실제 앞쪽으로 이동합니다"}'
            if step == "ad_staging_check":
                return '{"ready":true,"observation":"앞쪽 위치에서 실루엣이 온전히 보입니다"}'
            if step == "ad_layout":
                assert data["canvas_size"] == [1080, 1920]
                assert data["copy"]["headline"] == "포근한 겨울 저녁"
                assert data["revision_feedback"]["format_issue"] == str(issue)
                assert len(images) == (3 if staged else 2)
                answer = approved["design"] | {"subject_box": [67, 32, 20, 50]}
                if staged:
                    answer["boxes"] = answer["boxes"] | {"product_name": answer["boxes"]["headline"]}
                return json.dumps(answer)
            if step == "ad_composition":
                if calls.count(step) == 1:
                    if staged:
                        return '{"reason":"가장자리까지 상품을 키웁니다","revision":{"subject_box":[0,0,100,100]}}'
                    return '{"ready":false,"reason":"위쪽이 무의미하게 비어 있습니다","revision":null}'
                if staged:
                    assert data["layout_error"] == "전체를 보여 줄 상품이 사진 가장자리에 닿습니다"
                    return '{"ready":false,"reason":"제품명과 가격이 여전히 떨어져 있습니다","revision":null}'
                assert data["layout_error"] == "위쪽이 무의미하게 비어 있습니다"
                return '{"ready":true,"reason":"제목과 용기가 같은 축에 보입니다","revision":null}'
            if step == "ad_format_measured_layout":
                assert data["layout_error"] == "제품명과 가격이 여전히 떨어져 있습니다"
                assert len(images) == len(data["option_order"]) + 1
                return json.dumps({"placement": data["option_order"][0], "reason": "상품 옆에서 문구가 이어집니다"})
            if step == "ad_scene_direction":
                assert data["target_photo_aspect"] == [1080, 1920]
                assert "approved_design" not in data
                assert len(images) == (4 if staged else 3)
                assert data["approved_photo_cue"] == approved["photo_cue"]
                return '{"scene":"Warm lamp light guides the eye from the vessel.","photo_cue":"The rim casts a dark shadow."}'
            if step == "ad_photo_cue_check":
                assert len(images) == 2 and data["photo_cue"] == "The vessel reflects a strip of lamp light."
                return '{"visible":true,"observation":"용기에 조명이 반사됩니다","correction":null}'
            if step == "ad_final_layout":
                return '{"revision":null}'
            if step == "studio_staged_identity":
                assert len(images) == 3 and data["product_detail"] == "실제 용기"
                return json.dumps({"match": same_piece, "observation": "가까이서 상품 구조를 확인했습니다"})
            if step == "studio_repair_detail":
                return '{"target_box":null}'
            assert step == "studio_format_native_review"
            assert data["copy"]["price"] == "18,000원" and data["has_attempted_resize"]
            assert not data["has_staged_group"] and len(images) == 5
            assert images[-2].shape[:2] == (1920, 1080)
            assert images[-1].shape[:2] == (640, 360)
            return json.dumps({"ready": ready, "observation": "세로로 이어지는 빛을 확인했습니다"})

    def product_mask(_, pixels):
        result = np.zeros(pixels.shape[:2], np.float32)
        result[30:80, 65:92] = 1
        if staged and pixels.shape[:2] == (100, 100):
            result[30:56, 15:32] = 1
        return result

    from models.generative_ai.photo_retouch import product_mask as segmentation
    monkeypatch.setattr(segmentation, "product_mask", product_mask)
    monkeypatch.setattr(format_layout, "measure_texts", lambda probes, _: {row["role"]: (100, 35) for row in probes})
    output = tmp_path / "native"
    if not ready or not same_piece:
        reason = "새 구도를 확정하지" if same_piece else "강조한 상품이 원본과 다릅니다"
        with pytest.raises(ValueError, match=reason):
            studio_formats.rebuild_format(folder, output, "9:16", photo, fields, "", Agent(),
                                          {"mask": "mask", "generation": Model()}, (), issue)
        assert not output.exists()
    else:
        framing = studio_formats.rebuild_format(folder, output, "9:16", photo, fields, "", Agent(),
                                                {"mask": "mask", "generation": Model()}, (), issue)
        assert framing["method"] == "native"
        saved = load_layout(output)
        assert saved["format"] == "9:16" and saved["photo_cue"] == approved["photo_cue"]
        assert (output / "ad.png").exists()
        assert (output / "staging.png").exists() == staged
    stages = ["ad_format_staging", "ad_staging", "ad_staging_check"] if staged else []
    plans = ["ad_layout", "ad_layout"] if staged else ["ad_layout"]
    measured = ["ad_format_measured_layout"] if staged else []
    review = (["studio_staged_identity"] if staged else [])
    review += ["studio_format_native_review"] if same_piece else ["studio_repair_detail"]
    assert calls == [*stages, *plans, "ad_composition", "ad_composition", *measured, "ad_scene_direction", "photo",
                     "ad_photo_cue_check", "ad_final_layout", *review]


def test_native_detail_repair_recomposes_after_source_identity_passes(draft, monkeypatch):
    folder, _ = draft
    layout = load_layout(folder)
    comparisons = iter([(False, "상품의 작은 구조가 달라졌습니다"), (True, "원래 모양으로 돌아왔습니다")])
    monkeypatch.setattr(studio_formats, "check_staged_identity", lambda *args: next(comparisons))

    def repair(stage, plan, arguments, generation, ai):
        assert arguments["instruction"] == "상품의 작은 구조가 달라졌습니다"
        path = stage / plan["artwork"]["file"]
        with Image.open(path) as photo:
            fixed = photo.copy()
        fixed.paste("orange", (70, 70, 74, 74))
        fixed.save(path)

    monkeypatch.setattr(studio_formats, "repair_staged_detail", repair)
    assert studio_formats.restore_format_detail(folder, layout, None, {}) == "원래 모양으로 돌아왔습니다"
    with Image.open(folder / "ad.png") as image:
        assert image.getpixel((71, 71)) == (255, 165, 0)
        assert image.getpixel((75, 75)) == (0, 0, 255)


def test_detail_repair_keeps_photo_pixels_and_lettering_outside_the_box(tmp_path, draft):
    folder, _ = draft
    layout = load_layout(folder)
    layout["revision"] = {"staged_piece_box": [60, 40, 90, 70]}
    save_layout(layout, folder)
    Image.new("RGB", (60, 70), "orange").save(folder / "staged-piece.png")

    class Agent:
        def ask(self, step, images, request, *, model):
            assert step == "studio_repair_detail" and images[1].shape == (44, 44, 3)
            if "previous_plan" not in request:
                return '{"target_box":{"left":30,"top":30,"width":40,"height":40}}'
            return ('{"source_box":[20,20,50,50],"target_box":[30,30,40,40],'
                    '"instruction":"Restore the original broad rectangular upper part."}')

    class Model:
        def use_lora(self, weights, strength):
            assert weights is None and strength == 0

        def edit_with_reference(self, original, current, prompt, seed, steps):
            assert original.getpixel((0, 0)) == (255, 165, 0) and current.getpixel((0, 0)) == (0, 0, 255)
            assert "broad rectangular" in prompt
            return Image.new("RGB", current.size, "green")

    revised = tmp_path / "repaired"
    studio_edits.edit_version(folder, revised, "repair_staged_detail", {"instruction": "원래 부품 복원"},
                              lambda: {"generation": Model()}, ai=Agent())
    with Image.open(revised / "scene.png") as image:
        assert image.getbbox() == (0, 0, 100, 100)
        assert image.getpixel((74, 54)) == (0, 128, 0, 255)
        with Image.open(folder / "scene.png") as original:
            reverted = image.copy()
            target = (64, 44, 86, 66)
            reverted.paste(original.crop(target), target[:2])
            assert reverted.tobytes() == original.convert("RGBA").tobytes()
    assert (revised / "text_price.png").read_bytes() == (folder / "text_price.png").read_bytes()


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
            if step == "ad_final_composition_compare":
                assert data["original_ink"]["headline"][0] == 10 and data["revision_ink"]["headline"][0] == 27
                assert data["original_outer_space"] == data["revision_outer_space"] == {
                    "top": 10, "bottom": 20, "left": 10, "right": 15}
                assert images[0][10, 10].tolist() != images[1][10, 10].tolist()
                return '{"choice":"revision","reason":"사진 경계를 지나 제목과 제품이 이어집니다"}'
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


def test_studio_can_reframe_an_inset_photo_without_loading_a_generator(tmp_path, draft, monkeypatch):
    folder, fields = draft
    layout = load_layout(folder)
    layout["artwork"].update(y=10, h=85)
    layout["design"]["artwork_box"] = [0, 10, 100, 85]
    with Image.open(folder / "foreground.png") as original:
        mask = original.crop((0, 10, 100, 95))
    mask.save(folder / "foreground.png")
    Image.new("RGB", mask.size, "blue").save(folder / "scene.png")
    compose(layout, folder).save(folder / "ad.png")
    save_layout(layout, folder)
    monkeypatch.setitem(studio.FORMATS, "1:1", (100, 100))
    instruction = "사진 아래 빈 바탕을 줄여 주세요"

    def reframe(source, target, format_name, ai, expand, *, intent):
        assert format_name == "1:1" and expand is None and intent == instruction
        shutil.copytree(source, target)
        layout = load_layout(target)
        layout["artwork"].update(y=0, h=100)
        layout["design"]["artwork_box"] = [0, 0, 100, 100]
        full_mask = Image.new("L", (100, 100))
        full_mask.paste(mask, (0, 10))
        full_mask.save(target / "foreground.png")
        Image.new("RGB", full_mask.size, "blue").save(target / "scene.png")
        compose(layout, target).save(target / "ad.png")
        save_layout(layout, target)

    class Agent:
        def ask(self, step, images, request, *, model):
            data = json.loads(request.split("Input JSON follows:\n", 1)[1])
            if step == "studio_look":
                assert data["photo_boxes"]["v01"] == [0, 10, 100, 85]
                version = data["version_order"][-1]
                return json.dumps({"version": version, "finding": "layout", "observation": instruction})
            action = "reframe_photo" if "reframe_photo" in data["actions"] else "hold"
            arguments = {"instruction": instruction} if action == "reframe_photo" else {}
            return json.dumps({"action": action, "arguments": arguments, "reason": instruction})

    monkeypatch.setattr(studio, "adapt_format", reframe)
    result = studio.run_studio(Image.new("RGB", (100, 100)), fields, instruction, tmp_path / "reframe", Agent(),
                               draft=folder, edit_limit=2)
    assert result["version"] == "v02"
    assert load_layout(result["image"].parent)["texts"] == layout["texts"]


@pytest.mark.parametrize(("finding", "observation", "has_reference", "planned_layout"), [
    ("scene", "원본 촛불과 다른 연출이 보이지 않습니다", False, None),
    ("scene", "원본 촛불과 다른 연출이 보이지 않습니다", True, None),
    ("layout", "제목과 가격이 서로 어긋나 별개로 읽힙니다", True, None),
    ("product", "움직인 상품의 실제 구조가 바뀌었습니다", True, None),
    ("layout", "제목과 불빛이 떨어져 각각 읽힙니다", True, "불빛을 따라 제목을 읽습니다"),
])
def test_campaign_comparison_can_block_selection(tmp_path, draft, finding, observation, has_reference, planned_layout):
    folder, fields = draft
    if finding == "scene":
        Image.new("RGB", (100, 100), "green").save(folder / "composition.png")
    if finding == "product":
        layout = load_layout(folder)
        layout["revision"] = {"staged_piece_box": [20, 10, 60, 40], "staging_intent": "상품을 앞으로 옮깁니다"}
        save_layout(layout, folder)
        Image.new("RGB", (20, 20), "orange").save(folder / "staged-piece.png")
    if planned_layout:
        layout = load_layout(folder)
        layout["revision"] = {"layout_device": planned_layout}
        save_layout(layout, folder)
    references = [Image.new("RGB", (100, 100), "red")] if has_reference else []

    class Agent:
        def ask(self, step, images, request, *, model):
            if step == "studio_look":
                return '{"version":"v01","finding":"ready","observation":"중앙 정렬이 깔끔합니다"}'
            if step == "studio_staged_identity":
                assert finding == "product" and len(images) == 3 and images[1].shape == (50, 60, 3)
                return json.dumps({"match": False, "observation": observation})
            if step == "studio_campaign_check":
                assert finding != "product"
                has_study = finding == "scene"
                assert len(images) == 4 + has_study + has_reference and images[1][10, 10].tolist() == [1, 1, 1]
                assert images[2][10, 10].tolist() == [0, 0, 1]
                assert images[3].shape == (360, 360, 3)
                if has_study:
                    assert images[4][0, 0, 1] > 0 and images[4][0, 0, 0] == 0
                if has_reference:
                    assert images[-1][0, 0].tolist() == [1, 0, 0]
                data = json.loads(request.split("Input JSON follows:\n", 1)[1])
                assert data["has_source_study"] == has_study
                assert data["planned_layout"] == planned_layout
                assert data["rendered_text_bounds"]["headline"] == [10, 10, 13, 13]
                assert "포근한 겨울 저녁" in request
                return json.dumps({
                    "scene": {"treatment": "catalog" if finding == "scene" else "lighting",
                              "observation": observation if finding == "scene" else "촛불이 잘 보입니다"},
                    "layout": {"ready": finding != "layout" or bool(planned_layout),
                               "observation": observation if finding == "layout" else "문구가 연결됩니다",
                               **({"connection": "missing"} if planned_layout else {})},
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

    earlier_layout = load_layout(held)
    earlier_layout["pending_review"].pop("rules_hash")
    save_layout(earlier_layout, held)
    after_rule_change = tmp_path / "review-after-rule-change"
    studio.run_studio(Image.new("RGB", (100, 100)), fields, "", after_rule_change, Agent(), draft=held,
                      references=references, edit_limit=0)
    events = (after_rule_change / "events.jsonl").read_text(encoding="utf-8")
    assert ("staged_identity" if finding == "product" else "campaign_check") in events
    assert "resume_review" not in events

    held = after_rule_change / "versions/v01"
    with Image.open(held / "ad.png") as poster:
        changed = poster.copy()
    changed.putpixel((0, 0), (255, 0, 0))
    changed.save(held / "ad.png")
    fresh = tmp_path / "fresh-review"
    studio.run_studio(Image.new("RGB", (100, 100)), fields, "", fresh, Agent(), draft=held,
                      references=references, edit_limit=0)
    assert "resume_review" not in (fresh / "events.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize("action", ["change_viewpoint", "rebuild_campaign"])
def test_scene_action_keeps_improved_photo_for_layout_work(tmp_path, draft, monkeypatch, action):
    folder, fields = draft
    monkeypatch.setattr(studio, "available_viewpoints", lambda: {"high_angle": "하이 앵글"})
    ai = object()
    edits = []

    def ask(agent, step, task, data, images):
        assert agent is ai
        if step == "studio_look":
            return {"version": "v01", "finding": "scene", "observation": "윗면이 가려집니다"}
        if step == "studio_campaign_check":
            return {"scene": {"treatment": "feature_framing", "observation": "윗면이 보입니다"},
                    "layout": {"ready": False, "observation": "가격을 상품명 가까이 옮겨야 합니다"}}
        if action in data["actions"]:
            assert data["camera_viewpoints"] == {"high_angle": "하이 앵글"}
            arguments = ({"viewpoint": "high_angle", "instruction": "윗면이 보이도록 촬영"}
                         if action == "change_viewpoint" else {"intent": "제품의 윗면과 제목을 이어서 표현"})
            return {"action": action, "arguments": arguments, "reason": "윗면을 강조합니다"}
        return {"action": "hold", "arguments": {}, "reason": "촬영 결과를 검토합니다"}

    def remake(photo, facts, idea, models, output, **kwargs):
        assert facts == fields
        if action == "change_viewpoint":
            assert kwargs["scene_issue"] == "윗면이 가려집니다"
            assert kwargs["viewpoint"] == "high_angle" and kwargs["feedback"] == "윗면이 보이도록 촬영"
        else:
            assert kwargs["campaign_intent"] == "제품의 윗면과 제목을 이어서 표현" and "feedback" not in kwargs
        edits.append(kwargs["previous"])
        shutil.copytree(folder, output)

    monkeypatch.setattr(studio, "ask_stage", ask)
    monkeypatch.setattr(studio, "make_scene_ad", remake)
    output = tmp_path / "camera-viewpoint"
    result = studio.run_studio(Image.new("RGB", (100, 100)), fields, "", output, ai, dict,
                               draft=folder, edit_limit=1)
    assert edits == [output / "versions/v01"] and result["version"] == "v02"
    assert load_layout(result["image"].parent)["pending_review"]["finding"] == "layout"


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
    assert visible.getbbox() == (25, 25, 46, 76)

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


def test_cue_crop_compares_positions_but_keeps_the_visible_cue_protected():
    photo = Image.new("RGB", (160, 100), "blue")
    mask = Image.new("L", photo.size)
    mask.paste(255, (55, 25, 90, 85))
    mask.paste(255, (130, 0, 160, 25))

    def choose(previews, masks):
        assert set(previews) == {"start", "center", "end"}
        assert masks["start"].getbbox() == (55, 25, 90, 85)
        assert masks["end"].getbbox()[1] == 0
        return "start"

    _, visible = studio_formats.frame_photo(photo, mask, (100, 100), choose_position=choose,
                                            protected_box=(55, 25, 90, 85))
    assert visible.getbbox() == (55, 25, 90, 85)


def test_fit_preserves_clearance_for_a_complete_subject_but_allows_intentional_crops():
    photo = Image.new("RGB", (160, 100), "blue")
    mask = Image.new("L", photo.size)
    mask.paste(255, (50, 22, 110, 98))
    photo.paste("red", (50, 22, 110, 98))
    normal, bounds = studio_formats.frame_photo(photo, mask, (100, 100), keep_clearance=True)
    left, top, right, bottom = bounds.getbbox()
    assert bottom == 94 and right - left >= 55 and bottom - top >= 70
    assert normal.getpixel((50, 50)) == (255, 0, 0)
    assert normal.getpixel((50, 98))[2] > 245
    intentional, original_bounds = studio_formats.frame_photo(photo, mask, (100, 100))
    assert original_bounds.getbbox() == (20, 22, 80, 98)

    def reject(filled, preview, placed, method):
        assert placed == (0, 1, 100, 96) and method == "clearance"
        return "bottom"

    fallback, bounds = studio_formats.frame_photo(photo, mask, (100, 100), check_margin=reject, keep_clearance=True)
    assert fallback.tobytes() == intentional.tobytes() and bounds.tobytes() == original_bounds.tobytes()


def test_format_does_not_extend_a_cut_subject_into_blank_space():
    photo = Image.new("RGB", (100, 50), "blue")
    mask = Image.new("L", photo.size)
    mask.paste(255, (15, 0, 85, 50))
    with pytest.raises(studio_formats.FormatPhotoIssue, match="양쪽 사진 가장자리"):
        studio_formats.frame_photo(photo, mask, (100, 100))

    mask.paste(0, (0, 40, 100, 50))
    _, bounds = studio_formats.frame_photo(photo, mask, (100, 100),
                                           choose_position=lambda *_: pytest.fail("상품과 맞닿은 쪽에 여백을 둘 수 없습니다"))
    _, top, _, bottom = bounds.getbbox()
    assert top == 0 and bottom < 100


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


def test_format_margin_checks_rendered_ink_in_proportion_to_the_short_side():
    studio_formats.check_type_margin((1080, 1920), {"headline": [3, 2, 50, 17]})
    with pytest.raises(ValueError, match="headline 글자가 포스터 위쪽 가장자리에.*11픽셀"):
        studio_formats.check_type_margin((864, 1080), {"headline": [10, 1, 50, 17]})


def test_custom_format_rounds_absolute_text_edges_without_overlapping_rows():
    raw = {"boxes": {"product_name": [20, 13, 60, 4.5], "price": [20, 17.8, 60, 4.5]},
           "styles": {"headline": {"lines": [{"box": [13, 1, 74, 4.5]}, {"box": [20, 5.5, 60, 7]}]}}}
    result = studio_formats.selected_placement(raw, {}, {})
    assert result["boxes"] == {"product_name": [20, 13, 60, 5], "price": [20, 18, 60, 4]}
    assert result["styles"]["headline"]["lines"] == [{"box": [13, 1, 74, 5]}, {"box": [20, 6, 60, 6]}]


def test_product_name_and_price_cannot_occupy_opposite_ends_of_an_offer_line():
    ink = {"product_name": [24, 18, 45, 21], "price": [73, 18, 85, 21]}
    with pytest.raises(ValueError, match="상품명과 가격 사이의 빈 공간"):
        studio_formats.check_offer_spacing(ink, (1024, 1024))
    studio_formats.check_offer_spacing(ink | {"price": [48, 18, 60, 21]}, (1024, 1024))
    studio_formats.check_offer_spacing(ink | {"price": [30, 22, 42, 25]}, (1024, 1024))


def test_small_format_margin_repair_moves_the_entire_copy_without_changing_type():
    design = {"boxes": {"headline": [10, 0, 80, 15], "product_name": [10, 16, 80, 5], "price": [10, 22, 80, 4]},
              "styles": {"headline": {"lines": [{"box": [10, 0, 80, 7], "size": 5},
                                                {"box": [10, 8, 80, 7], "size": 6}]},
                         "price": {"size": 3}}}
    ink = {"headline": [12, 1, 60, 14], "product_name": [12, 16, 38, 20], "price": [12, 22, 20, 25]}
    moved = studio_formats.move_copy_inside(design, ink, (1024, 1024))
    assert moved["boxes"]["headline"] == [10, 2, 80, 15]
    assert moved["boxes"]["price"] == [10, 24, 80, 4]
    assert moved["styles"]["headline"]["lines"][1] == {"box": [10, 10, 80, 7], "size": 6}
    assert moved["styles"]["price"] == design["styles"]["price"]


def test_format_options_use_measured_sizes_and_keep_clear_of_the_product(monkeypatch):
    style = {"align": "left", "max_font_size": 8}
    rows = [style | {"role": "headline_line_1"}, style | {"role": "headline_line_2"}]
    source_style = {"size": 6, "lines": [{"box": [8, 5, 35, 9]}, {"box": [8, 15, 40, 11]}]}
    layout = {"size": [100, 100], "texts": [style | {"role": "headline", "lines": rows},
              style | {"role": "product_name"}, style | {"role": "price"}],
              "design": {"focus": "price", "boxes": {"headline": [8, 5, 40, 22]},
                         "styles": {"headline": source_style, "product_name": {}, "price": {}}}}
    mask = np.zeros((100, 100), dtype=bool)
    mask[34:70, 20:87] = True

    def measure(texts, size):
        assert size == (100, 100) and all(text["max_font_size"] == 8 for text in texts)
        return dict(zip((text["role"] for text in texts), [(24, 7), (32, 9), (12, 7), (14, 7)], strict=True))

    monkeypatch.setattr(format_layout, "measure_texts", measure)
    dimensions = format_layout.format_measurements(layout, (100, 100))
    options = format_layout.position_options(layout, (100, 100), mask, [20, 34, 87, 70], dimensions)
    chosen = studio_formats.selected_placement({"placement": "headline_above_offer_below"}, options, layout["design"])
    assert chosen["boxes"]["headline"][2:] == [35, 23]
    assert chosen["boxes"]["headline"][1] + 23 < 34
    assert chosen["boxes"]["price"][1] > 70
    assert chosen["styles"]["headline"]["lines"][0]["box"][2:] == [27, 10]
    assert source_style["lines"][0]["box"] == [8, 5, 35, 9]
    for option in options.values():
        for x, y, w, h in option["boxes"].values():
            assert not mask[y:y+h, x:x+w].any()


def test_requested_format_repair_can_use_custom_layout_when_measured_options_exist(tmp_path, draft, monkeypatch):
    folder, _ = draft
    monkeypatch.setattr(studio_formats, "format_measurements", lambda *_: None)
    monkeypatch.setattr(studio_formats, "position_options", lambda *_: {"measured": {}})
    intent = "제목을 상품 옆의 여백으로 옮겨 주세요"

    class Agent:
        def ask(self, step, images, request, *, model):
            data = json.loads(request.split("Input JSON follows:\n", 1)[1])
            if step == "studio_reframe":
                assert data["intent"] == intent and "measured" in data["placement_options"]
                revised = data["approved_design"]
                revised["boxes"]["headline"] = [10, 20, 45, 10]
                return json.dumps(revised)
            assert step == "studio_format_reading"
            return '{"finding":"ready","headline_touches_product":false,"observation":"상품 옆에 제목이 보입니다"}'

    output = tmp_path / "reframed"
    studio_formats.adapt_format(folder, output, "thumbnail", Agent(), intent=intent)
    assert load_layout(output)["design"]["boxes"]["headline"] == [10, 20, 45, 10]


def test_format_price_stays_secondary_except_when_the_price_is_the_focus(monkeypatch):
    styles = {"headline": {"size": 5, "lines": [{"size": 4}, {"size": 8}]},
              "product_name": {"size": 6}, "price": {"size": 7, "font": "noto_sans_kr_black"}}
    assert format_layout.format_type_hierarchy(styles, "product")["price"] == styles["price"] | {"size": 4.8}
    assert format_layout.format_type_hierarchy(styles, "product_name")["price"]["size"] == pytest.approx(3.6)
    assert format_layout.format_type_hierarchy(styles, "price") is styles
    assert format_layout.supporting_price_size("headline", 7, 2.5, 2) == 2
    source = {"size": (1000, 1000), "texts": [{"role": "headline", "max_font_size": 80},
              {"role": "product_name", "max_font_size": 60}, {"role": "price", "max_font_size": 70}],
              "design": {"focus": "product"}}
    monkeypatch.setattr(format_layout, "measure_texts",
                        lambda probes, _: {row["role"]: (100, row["max_font_size"]) for row in probes})
    dimensions, sizes = format_layout.format_measurements(source, (1280, 720))
    assert dimensions["price"][1] < dimensions["headline"][1]
    source["design"]["focus"] = "price"
    _, featured_sizes = format_layout.format_measurements(source, (1280, 720))
    assert sizes["price"] < featured_sizes["price"] and sizes["headline"] == featured_sizes["headline"]


def test_expands_only_the_chosen_photo_placement():
    photo = Image.new("RGB", (100, 100), "blue")
    mask = Image.new("L", photo.size)
    mask.paste(255, (7, 25, 93, 76))
    expanded = []

    def choose(previews, masks):
        assert list(previews) == ["top", "center", "bottom"]
        assert all(preview.size == (50, 100) for preview in previews.values())
        assert masks["top"].getbbox()[1] < masks["bottom"].getbbox()[1]
        return "bottom"

    def fill(frame, margin):
        assert margin[:39].all() and margin[92:].all() and not margin[39:92].any()
        expanded.append(frame)
        return Image.new("RGB", frame.size, "green")

    result, visible = studio_formats.frame_photo(photo, mask, (50, 100), fill, choose)
    assert len(expanded) == 1 and visible.getbbox() == (2, 52, 48, 79)
    assert result.crop((0, 39, 50, 92)).tobytes() == expanded[0].crop((0, 39, 50, 92)).tobytes()


def test_format_removes_the_failed_margin_without_repainting_the_photo():
    photo = Image.new("RGB", (100, 100), "blue")
    mask = Image.new("L", photo.size)
    mask.paste(255, (0, 25, 100, 76))
    reviews = []

    def expand(frame, margin):
        return Image.new("RGB", frame.size, "red" if margin[0, 0] else "green")

    def review(filled, _, placed, method):
        reviews.append((placed, method))
        assert filled.crop(placed).tobytes() == Image.new("RGB", (50, 50), "blue").tobytes()
        if method == "reflected":
            assert filled.getpixel((0, 0)) == (0, 0, 255) and filled.getpixel((0, 99)) == (255, 0, 0)
        return None if method == "relocated" else "top"

    result, visible = studio_formats.frame_photo(photo, mask, (50, 100), expand,
                                               choose_position=lambda *_: "center", check_margin=review)
    assert reviews == [((0, 25, 50, 75), "initial"), ((0, 25, 50, 75), "reflected"), ((0, 0, 50, 50), "relocated")]
    assert result.getpixel((0, 0)) == (0, 0, 255) and result.getpixel((0, 99)) == (0, 128, 0)
    assert visible.getbbox()[3] <= 50


def test_reflected_margin_uses_the_background_without_repeating_the_product():
    photo = Image.new("RGB", (20, 40), "blue")
    photo.paste("red", (0, 15, 20, 30))
    mask = Image.new("L", photo.size)
    mask.paste(255, (0, 15, 20, 30))
    frame = Image.new("RGB", (20, 85), "black")
    bounds = (0, 10, 20, 50)
    frame.paste(photo, bounds[:2])
    result = studio_formats.reflect_margin(frame, photo, mask, bounds, "bottom")
    assert result.crop((0, 50, 20, 85)).getcolors() == [(700, (0, 0, 255))]
    assert result.crop((0, 0, 20, 50)).tobytes() == frame.crop((0, 0, 20, 50)).tobytes()


def test_reflected_margins_leave_clean_corners_when_expanding_both_axes():
    photo = Image.new("RGB", (70, 80), "white")
    subject = Image.new("L", photo.size)
    subject.paste(255, (20, 25, 50, 55))
    filled, placed = studio_formats.extend_edges(photo, (100, 100), 0.5)
    for side in ("left", "right", "top", "bottom"):
        filled = studio_formats.reflect_margin(filled, photo, subject, placed, side)
    assert filled.getextrema() == ((255, 255),) * 3


def test_wide_crop_keeps_available_space_beyond_the_product():
    photo = Image.new("RGB", (100, 100), "white")
    mask = Image.new("L", photo.size)
    mask.paste(255, (20, 32, 80, 74))
    _, framed = studio_formats.frame_photo(photo, mask, (100, 50))
    assert framed.getbbox() == (20, 4, 80, 46)


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
