import json

import numpy as np
from PIL import Image
import pytest

from models.generative_ai.ad_layers import scene_ad
from models.generative_ai.ad_layers.layers import compose
from models.generative_ai.ad_layers.text_render import page_html
from models.generative_ai.photo_retouch import product_mask


def test_scene_ad_plans_canvas_and_exact_copy_before_generation(tmp_path, monkeypatch):
    calls = []
    photo = Image.new("RGB", (70, 60), "orange")
    reference = Image.new("RGB", (70, 60), "white")
    fields = {"product_name": "수분 세럼", "price_krw": 32000}
    correction = "Show the cap held visibly by a hand."
    actual_light = "Grazing light from the left defines the small drop under the cap."
    cue = "The near glass edge catches an amber highlight."
    cue_correction = "Move the side light to reveal the near glass edge."
    exercise_cue_retry = True
    reject_cue = False
    campaign_intent = "빛줄기와 제목을 같은 방향으로 연결합니다"
    expect_correction = False
    delegated_framing = False

    def mask(model, pixels):
        assert model == "mask"
        foreground = np.zeros(pixels.shape[:2], np.float32)
        if pixels.shape[:2] == (60, 70):
            foreground[18:46, 20:50] = 1
        else:
            assert pixels[0, 0, 2] == 1
            foreground[20:75, 15:47] = 1
        return foreground

    class Model:
        def use_lora(self, weights, strength):
            assert weights is None and strength == 0

        def edit(self, reference, prompt, seed, steps, *, size):
            assert size == (512, 1024)
            if reject_cue and "ad_photo_cue_check" in calls:
                assert reference.size == photo.size and reference.tobytes() == photo.tobytes()
                assert "The reference identifies the original item" in prompt
                assert "Follow the reference's product placement" not in prompt and cue in prompt
                calls.append("qwen")
                return Image.new("RGB", size, "blue")
            assert reference.size == size
            assert reference.getpixel((0, 0)) == (128, 128, 128)
            assert reference.getpixel((243, 460)) == photo.getpixel((30, 28))
            is_revision = "ad_cycle_critique" in calls
            assert ("Narrow studio beam" if is_revision else "Amber reflections") in prompt
            assert "Shoot near the subject's height" in prompt
            assert ("Show the amber edge in side light." in prompt) == (not delegated_framing)
            assert (correction in prompt) == (expect_correction and not delegated_framing)
            assert ("Keep the same close framing" in prompt) == expect_correction
            if expect_correction:
                assert reference.getpixel((100, 460)) == (128, 128, 128)
            assert "부드러운 아침" not in prompt
            if exercise_cue_retry and "ad_photo_cue_check" in calls:
                assert prompt.startswith(f"Priority reshoot correction, replacing any conflicting placement instruction: {cue_correction}")
            assert cue in prompt
            calls.append("qwen")
            return Image.new("RGB", size, (160, 40, 255) if is_revision else (0, 0, 255))

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
            if step == "ad_art_direction":
                assert len(images) == 2 and "부드러운 아침" in prompt
                return json.dumps({"theme": "studio", "angle": "level", "shot_scale": "whole",
                                   "lighting": actual_light if expect_correction else "transmitted", "evidence": "앰버색 병",
                                   "photo_device": "Show the amber edge in side light."})
            if step == "ad_source_framing":
                assert len(images) == 3 and images[0].shape == (28, 30, 3) and images[1].shape == (60, 70, 3)
                if delegated_framing:
                    assert "앰버 유리가 묻힙니다" in prompt and correction not in prompt
                crop = [33, 10, 40, 80] if expect_correction else None
                return json.dumps({"source_crop": crop, "reason": "병의 실제 형태를 확인합니다"})
            if step == "ad_source_framing_check":
                assert expect_correction and images[0].shape == images[1].shape == (512, 512, 3)
                assert not np.array_equal(images[0], images[1])
                return '{"choice":"detail","reason":"앰버 유리의 빛이 확대되어 보입니다"}'
            if step == "ad_cycle_critique":
                data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
                assert images[0][20, 15, 2] == 1
                critique = {"weakness": "평평한 조명", "hypothesis": "A narrow beam through the glass"}
                if "editor_intent" in data:
                    assert len(images) == 3 and data["editor_intent"] == campaign_intent
                    critique.update(layout_device="빛줄기와 제목을 나란히", visible_result="빛이 제목으로 이어짐")
                else:
                    assert len(images) == 4 and "병의 하단이 잘렸습니다" in prompt
                return json.dumps(critique)
            if step == "ad_scene_direction":
                data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
                assert data["design"]["artwork_box"] == [0, 0, 50, 100]
                assert "vertical layers" in data["angle_guide"] and data["art_direction_plan"]["angle"] == "level"
                assert (actual_light if expect_correction else "diffused light behind") in data["lighting_guide"]
                assert len(images) == 2 and images[1].shape == (1024, 1024, 3) and "previous_art_direction" not in data
                if expect_correction:
                    assert data["requested_scene_change"] == correction
                    assert "deliberate detail crop" in data["framing_guide"]
                if delegated_framing:
                    assert "Omit any instruction to keep the entire product" in prompt
                if "hypothesis" in data:
                    assert data["hypothesis"] == "A narrow beam through the glass" and data["idea"] == "가벼운 아침 연출"
                    if "layout_device" in data:
                        assert data["visible_result"] == "빛이 제목으로 이어짐"
                    return json.dumps({"scene": "Narrow studio beam", "photo_cue": cue})
                return json.dumps({"scene": "Amber reflections", "photo_cue": cue})
            if step == "ad_photo_cue_check":
                assert cue in prompt and len(images) == 2 and images[1].shape == (1024, 512, 3)
                if reject_cue or exercise_cue_retry and calls.count(step) == 1:
                    return json.dumps({"visible": False, "observation": "병 가장자리에 빛이 없습니다",
                                       "correction": cue_correction, "reposition": reject_cue})
                return '{"visible":true,"observation":"유리 가장자리에 빛이 보입니다","correction":null}'
            if step == "ad_composition":
                assert len(images) == 3 and images[0].shape == (1024, 1024, 3)
                data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
                assert data["copy"]["headline"] == "부드러운 아침"
                assert set(data["rendered_text_bounds"]) == {"headline", "product_name", "price"}
                return ('{"reason":"병 위치를 옮깁니다",'
                        '"revision":{"subject_box":[15,20,70,50],"crop":"preserve","source_crop":null}}')
            if step == "ad_composition_compare":
                assert len(images) == 3 and not np.array_equal(images[0], images[1])
                return '{"choice":"original","reason":"기존 위치에서 제목이 더 자연스럽게 읽힙니다"}'
            if step == "ad_final_layout":
                assert len(images) == 3 and images[1][0, 0, 2] == 1
                data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
                assert "product_name 글자 레이어가 실제 제품을 가립니다" in data["layout_error"]
                assert set(data["rendered_text_bounds"]) == {"headline", "product_name", "price"}
                boxes = {"headline": [55, 10, 40, 30], "product_name": [55, 45, 40, 15], "price": [55, 65, 40, 15]}
                if expect_correction:
                    if calls.count(step) == 1:
                        return '{"revision":null}'
                    assert "배치 수정이 빠졌습니다" in data["layout_error"]
                    revised = data["design"] | {"boxes": boxes, "accent": None, "edge": "blend",
                                                  "artwork_box": [0, 0, 100, 100]}
                    revised["styles"]["headline"].update(size=6.3, depth="front")
                    return json.dumps(revised)
                return json.dumps({"reason": "상품명을 제품 옆으로 이동", "revision": {"boxes": boxes, "accent": None}})
            assert step == "ad_layout" and images[0][0, 0, 2] == 0
            assert len(images) == (3 if expect_correction else 2)
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            assert data["copy"] == {"headline": "부드러운 아침", "product_name": "수분 세럼", "price": "32,000원"}
            assert "one axis" in data["art_direction_plan"]["layout_guide"]
            if "revision_feedback" in data:
                assert "scene" not in data["brief"] and data["idea"] == "가벼운 아침 연출"
            if expect_correction:
                assert data["revision_feedback"]["requested_scene_change"] == correction and images[1][20, 15, 2] == 1
            boxes = {"headline": [55, 10, 40, 30], "product_name": [2, 2, 40, 15], "price": [55, 65, 40, 15]}
            if "layout_error" not in data:
                boxes["headline"] = [55, 30, 40, 40]
            else:
                assert "겹쳤습니다" in data["layout_error"]
            style = {"font": "noto_sans_kr", "align": "left", "color": "#ffffff", "size": 5, "max_lines": 1}
            return json.dumps({"concept": "Light joins image and type", "focus": "headline", "background": "#251d16",
                               "accent": {"shape": "ellipse", "box": [35, 10, 55, 55], "color": "#9e6c4b"},
                               "artwork_box": [0, 0, 50, 100], "subject_box": [10, 20, 75, 50],
                               "edge": "framed", "crop": "preserve",
                               "art_direction": "Amber reflections",
                               "boxes": boxes, "styles": {role: style for role in boxes}})

    def render(texts, size, folder):
        calls.append("render")
        assert size == (1024, 1024)
        for text in texts:
            ink = Image.new("RGBA", (text["w"], text["h"]))
            ink.paste("white", (0, 0, 8, 8))
            ink.save(folder / text["file"])

    monkeypatch.setattr(product_mask, "product_mask", mask)
    monkeypatch.setattr(scene_ad, "render_texts", render)
    folder = tmp_path / "result"
    models = {"ai": Interpreter(), "generation": Model(), "mask": "mask"}
    layout = scene_ad.make_scene_ad(photo, fields, "", models, folder, references=[reference])
    assert calls == ["ad_plan", "ad_direction", "ad_headline", "ad_art_direction", "ad_source_framing", "ad_layout",
                     "ad_layout", "render", "ad_composition", "render", "ad_composition_compare", "ad_scene_direction", "qwen",
                     "ad_photo_cue_check", "qwen", "ad_photo_cue_check", "ad_final_layout", "render"]
    assert layout["photo_cue"] == cue and [check["visible"] for check in layout["photo_cue_checks"]] == [False, True]
    exercise_cue_retry = False
    assert layout["design"]["accent"] is None
    assert layout["art_direction_plan"]["evidence"] == "앰버색 병"
    assert layout["final_review_reason"] == "상품명을 제품 옆으로 이동"
    assert (folder / "before-typesetting.png").is_file()
    assert (folder / "composition.png").is_file()
    assert not (folder / "before-composition.png").exists()
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
    calls.clear()
    accepted = scene_ad.make_scene_ad(photo, fields, "", models, tmp_path / "accepted",
                                      references=[reference], proposal=layout)
    assert calls == ["render", "qwen", "ad_photo_cue_check"]
    assert accepted["design"] == layout["design"] and accepted["final_review_reason"] is None
    reject_cue = True
    calls.clear()
    unapproved = tmp_path / "unapproved"
    with pytest.raises(ValueError, match="사진에 의도한 연출이 보이지 않습니다"):
        scene_ad.make_scene_ad(photo, fields, "", models, unapproved, proposal=layout)
    assert calls == ["render", "qwen", "ad_photo_cue_check", "qwen", "ad_photo_cue_check"]
    assert not unapproved.exists()
    reject_cue = False
    calls.clear()
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    (rejected / "ad.png").write_bytes((folder / "ad.png").read_bytes())
    rejected_layout = layout | {"revision": {"visual_review": "병의 하단이 잘렸습니다"}}
    (rejected / "layout.json").write_text(json.dumps(rejected_layout), encoding="utf-8")
    revision = scene_ad.make_scene_ad(photo, fields, "가벼운 아침 연출", models, tmp_path / "revision", references=[reference],
                                      previous=folder, rejected=rejected)
    assert calls == ["ad_cycle_critique", "ad_art_direction", "ad_source_framing", "ad_layout", "ad_layout", "render", "ad_composition",
                     "render", "ad_composition_compare", "ad_scene_direction", "qwen", "ad_photo_cue_check", "ad_final_layout", "render"]
    assert revision["revision"]["weakness"] == "평평한 조명"
    assert revision["design"]["art_direction"] == "Narrow studio beam"
    assert revision["brief"]["headline"] == layout["brief"]["headline"]

    calls.clear()
    rebuilt = scene_ad.make_scene_ad(photo, fields, "가벼운 아침 연출", models, tmp_path / "rebuilt",
                                     references=[reference], previous=folder, campaign_intent=campaign_intent)
    assert rebuilt["revision"]["layout_device"] == "빛줄기와 제목을 나란히"
    assert [item["text"] for item in rebuilt["texts"]] == [item["text"] for item in layout["texts"]]

    calls.clear()
    expect_correction = True
    targeted = scene_ad.make_scene_ad(photo, fields, "가벼운 아침 연출", models, tmp_path / "targeted",
                                      references=[reference], previous=folder, feedback=correction)
    assert "ad_cycle_critique" not in calls
    assert targeted["revision"]["requested_scene_change"] == correction
    assert correction not in targeted["design"]["art_direction"] and correction in targeted["scene_prompt"]
    assert targeted["brief"]["headline"] == layout["brief"]["headline"]
    assert calls.count("ad_final_layout") == 2 and calls.count("qwen") == 1
    assert targeted["artwork"] == layout["artwork"]
    assert targeted["design"]["source_crop"] == targeted["source_framing"]["source_crop"] == [33, 10, 40, 80]
    headline = next(text for text in targeted["texts"] if text["role"] == "headline")
    assert headline["max_font_size"] == 65 and "depth" not in headline
    assert targeted["design"]["edge"] == "blend"
    with Image.open(tmp_path / "targeted" / "scene.png") as scene:
        assert scene.getpixel((15, 20)) == (0, 0, 255, 255)
        assert scene.getpixel((511, 20)) == (0, 0, 255, 0)

    calls.clear()
    delegated_framing = True
    delegated = scene_ad.make_scene_ad(photo, fields, "가벼운 아침 연출", models, tmp_path / "delegated",
                                       references=[reference], previous=folder, feedback=correction,
                                       scene_issue="앰버 유리가 묻힙니다")
    assert delegated["design"]["source_crop"] == [33, 10, 40, 80]
    assert correction not in delegated["scene_prompt"]


def test_cropped_side_stays_at_the_photo_edge_in_model_reference():
    product = Image.new("RGBA", (20, 40), "orange")
    crop = [0, 50, 100, 50]
    detail = scene_ad.crop_subject(product, {"source_crop": crop})
    picture_size = (40, 80)
    subject_box = [0, 0, 100, 100]

    assert scene_ad.frame_subject(product, picture_size, subject_box).getbbox() == (0, 0, 40, 80)
    assert scene_ad.frame_subject(detail, picture_size, subject_box, crop).getbbox() == (0, 0, 40, 40)
    reference = scene_ad.product_reference(product, detail, picture_size, subject_box, crop)
    assert reference.getpixel((10, 0)) == (255, 165, 0)
    assert reference.getpixel((10, 79)) == (128, 128, 128)


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


def test_headline_can_tuck_behind_the_subject_without_repainting_it(tmp_path):
    size = (30, 30)
    Image.new("RGB", size, "blue").save(tmp_path / "scene.png")
    Image.new("RGB", size, "black").save(tmp_path / "background.png")
    mask = np.zeros((30, 30), dtype=bool)
    mask[10, 12] = True
    Image.fromarray(mask).save(tmp_path / "foreground.png")
    ink = Image.new("RGBA", size)
    ink.paste("white", (2, 10, 13, 11))
    ink.save(tmp_path / "title.png")
    layout = {"size": list(size), "background": {"file": "background.png"},
              "artwork": {"file": "scene.png", "x": 0, "y": 0, "w": 30, "h": 30},
              "texts": [{"role": "headline", "file": "title.png", "x": 0, "y": 0, "w": 30, "h": 30,
                         "depth": "behind_subject"}]}
    assert scene_ad.check_text_overlap(mask, layout, tmp_path) is True
    assert scene_ad.check_text_overlap(np.zeros_like(mask), layout, tmp_path) is False
    rendered = compose(layout, tmp_path)
    assert rendered.getpixel((12, 10)) == (0, 0, 255)
    assert rendered.getpixel((11, 10)) == (255, 255, 255)
    assert Image.open(tmp_path / "title.png").getpixel((12, 10)) == (255, 255, 255, 255)
    mask[10, 10:13] = True
    with pytest.raises(ValueError, match="제목이 너무 많아"):
        scene_ad.check_text_overlap(mask, layout, tmp_path)

    mask[:] = False
    ink.paste("white", (2, 10, 13, 15))
    ink.paste("white", (17, 16, 19, 18))
    ink.save(tmp_path / "title.png")
    mask[16:18, 17:19] = True
    with pytest.raises(ValueError, match="제목이 너무 많아"):
        scene_ad.check_text_overlap(mask, layout, tmp_path)


def test_foreground_can_include_a_hand_touching_the_frame(monkeypatch):
    mask = np.zeros((30, 30), dtype=np.float32)
    mask[0:8, 21:30] = 1  # 제품을 보여 주는 손
    mask[12:25, 14:20] = 1  # 화면 안쪽에 있는 제품
    monkeypatch.setattr(product_mask, "product_mask", lambda *_: mask)
    assert np.array_equal(scene_ad.find_foreground(Image.new("RGB", (30, 30)), None, 0.5), mask > 0.5)


def test_graphic_goes_between_full_bleed_photo_and_type(tmp_path, monkeypatch):
    def render(texts, size, folder):
        for text in texts:
            Image.new("RGBA", (text["w"], text["h"]), "white").save(folder / text["file"])

    monkeypatch.setattr(scene_ad, "render_texts", render)
    boxes = {"headline": [20, 20, 10, 10], "product_name": [50, 40, 10, 10], "price": [50, 60, 10, 10]}
    style = {"font": "noto_sans_kr", "align": "left", "color": "#ffffff", "size": 5, "max_lines": 1}
    answer = {"concept": "Extend a color from the photo", "art_direction": "Warm window light",
              "focus": "product", "crop": "preserve", "edge": "framed", "background": "#ffffff",
              "artwork_box": [0, 0, 100, 100], "accent": {"shape": "rectangle", "layer": "overlay",
              "box": [10, 10, 25, 25], "color": "#ff0000"}, "boxes": boxes,
              "styles": dict.fromkeys(boxes, style)}
    layout = scene_ad.design_layers(answer, dict.fromkeys(boxes, "광고"), (100, 100), ["noto_sans_kr"], tmp_path)
    Image.new("RGB", (100, 100), "blue").save(tmp_path / "scene.png")
    poster = scene_ad.compose(layout, tmp_path)
    assert poster.getpixel((5, 5)) == (0, 0, 255)
    assert poster.getpixel((15, 15)) == (255, 0, 0)
    assert poster.getpixel((25, 25)) == (255, 255, 255)
    foreground = np.zeros((100, 100), dtype=bool)
    scene_ad.check_text_overlap(foreground, layout, tmp_path)
    foreground[15, 15] = True
    with pytest.raises(ValueError, match="색면 레이어가 실제 제품을 가립니다"):
        scene_ad.check_text_overlap(foreground, layout, tmp_path)

    answer["accent"]["layer"] = "behind_subject"
    layout = scene_ad.design_layers(answer, dict.fromkeys(boxes, "광고"), (100, 100), ["noto_sans_kr"], tmp_path)
    Image.fromarray(foreground).save(tmp_path / "foreground.png")
    scene_ad.check_text_overlap(foreground, layout, tmp_path)
    poster = scene_ad.compose(layout, tmp_path)
    assert poster.getpixel((15, 15)) == (0, 0, 255)
    assert poster.getpixel((16, 15)) == (255, 0, 0)
    assert poster.getpixel((25, 25)) == (255, 255, 255)
    photo = Image.new("RGB", (100, 100), "white")
    photo.putpixel((16, 15), (128, 128, 128))
    photo.save(tmp_path / "scene.png")
    poster = scene_ad.compose(layout, tmp_path)
    assert poster.getpixel((16, 15)) == (128, 0, 0)
    assert poster.getpixel((17, 15)) == (255, 0, 0)
    with pytest.raises(ValueError, match="바깥에서도 보여야"):
        scene_ad.check_text_overlap(np.zeros_like(foreground), layout, tmp_path)


def test_designed_line_break_reaches_html_renderer():
    boxes = {"headline": [0, 0, 50, 40], "product_name": [0, 50, 50, 15], "price": [0, 70, 50, 15]}
    style = {"font": "noto_sans_kr", "align": "left", "color": "#ffffff", "size": 5, "max_lines": 2,
             "break_after": 3}
    copy = {"headline": "패티 두 장이면 든든 <맛>", "product_name": "햄버거", "price": "3,000원"}
    support = {key: value for key, value in style.items() if key != "break_after"} | {"max_lines": 1}
    styles = {"headline": style, "product_name": support, "price": support}
    layer = scene_ad.layout_texts({"boxes": boxes, "styles": styles}, copy, (1024, 1024), ["noto_sans_kr"])
    assert layer[0]["text"] == copy["headline"]
    assert "패티 두 장이면<br>든든 &lt;맛&gt;" in page_html(layer, (1024, 1024))
    layer[0]["text"] = "새 제목"
    assert "<span>새 제목</span>" in page_html(layer, (1024, 1024))
    style["emphasis"] = {"phrase": "장이면 든든", "color": "#ffa040"}
    layer = scene_ad.layout_texts({"boxes": boxes, "styles": styles}, copy, (1024, 1024), ["noto_sans_kr"])
    assert layer[0]["text"] == copy["headline"]
    assert ('패티 두 <span style="color:#ffa040">장이면</span><br>'
            '<span style="color:#ffa040">든든</span> &lt;맛&gt;') in page_html(layer, (1024, 1024))
    style["emphasis"] = {"phrase": "장이", "color": "#ffa040"}
    with pytest.raises(ValueError, match="강조는 제목"):
        scene_ad.layout_texts({"boxes": boxes, "styles": styles}, copy, (1024, 1024), ["noto_sans_kr"])
    style["break_after"] = 5
    with pytest.raises(ValueError, match="줄바꿈 위치"):
        scene_ad.layout_texts({"boxes": boxes, "styles": styles}, copy, (1024, 1024), ["noto_sans_kr"])


def test_split_headline_keeps_copy_and_checks_each_visible_line(tmp_path):
    boxes = {"headline": [0, 0, 90, 52], "product_name": [0, 60, 50, 15], "price": [0, 80, 50, 15]}
    style = {"font": "noto_sans_kr", "align": "left", "color": "#ffffff", "size": 10, "max_lines": 1}
    lines = [{"box": [10, 10, 60, 12], "size": 6, "text": "따뜻한 빛으로"},
             {"box": [25, 35, 60, 12], "color": "#ffa040"}]
    headline = style | {"max_lines": 2, "break_after": 2, "lines": lines,
                        "emphasis": {"phrase": "빛으로 저녁을", "color": "#e04030"}}
    answer = {"boxes": boxes, "styles": {"headline": headline, "product_name": style, "price": style}}
    copy = {"headline": "따뜻한 빛으로 저녁을 감싸요", "product_name": "조명", "price": "30,000원"}
    assert scene_ad.headline_breaks(copy["headline"])[1] == {
        "break_after": 2, "first_line": "따뜻한 빛으로", "second_line": "저녁을 감싸요"}
    texts = scene_ad.layout_texts(answer, copy, (100, 100), ["noto_sans_kr"])
    first, second = texts[0]["lines"]
    assert texts[0]["text"] == copy["headline"]
    assert (first["text"], second["text"]) == ("따뜻한 빛으로", "저녁을 감싸요")
    assert [line["max_font_size"] for line in (first, second)] == [6, 10]
    html = page_html([first, second], (100, 100))
    assert '따뜻한 <span style="color:#e04030">빛으로</span>' in html
    assert '<span style="color:#e04030">저녁을</span> 감싸요' in html
    for text in texts:
        for line in text.get("lines", (text,)):
            image = Image.new("RGBA", (line["w"], line["h"]))
            image.paste(line["color"], (0, 0, 3, 3))
            image.save(tmp_path / line["file"])
    Image.new("RGB", (100, 100), "black").save(tmp_path / "background.png")
    Image.new("RGBA", (100, 100)).save(tmp_path / "scene.png")
    layout = {"size": [100, 100], "texts": texts, "background": {"file": "background.png"},
              "artwork": {"file": "scene.png", "x": 0, "y": 0, "w": 100, "h": 100}}
    rendered = scene_ad.compose(layout, tmp_path)
    assert rendered.getpixel((11, 11)) == (255, 255, 255)
    assert rendered.getpixel((26, 36)) == (255, 160, 64)
    bounds = scene_ad.rendered_text_bounds(layout, tmp_path)
    assert bounds["headline"] == [10, 10, 28, 38] and bounds["headline_line_2"] == [25, 35, 28, 38]
    mask = np.zeros((100, 100), dtype=bool)
    mask[35, 25] = True
    with pytest.raises(ValueError, match="headline_line_2"):
        scene_ad.check_text_overlap(mask, layout, tmp_path)
    lines[0]["text"] = "다른 문구"
    with pytest.raises(ValueError, match="확정된 원문"):
        scene_ad.layout_texts(answer, copy, (100, 100), ["noto_sans_kr"])
    del lines[0]["text"]
    lines[1]["box"] = [25, 51, 60, 12]
    with pytest.raises(ValueError, match="제목 전체 영역"):
        scene_ad.layout_texts(answer, copy, (100, 100), ["noto_sans_kr"])


@pytest.mark.parametrize("blocked", [False, True])
def test_split_title_closes_the_visible_gap_without_covering_the_product(tmp_path, blocked):
    lines = [
        {"role": "headline_line_1", "file": "first.png", "x": 10, "y": 10, "w": 50, "h": 20},
        {"role": "headline_line_2", "file": "second.png", "x": 10, "y": 31, "w": 50, "h": 25},
    ]
    for line, ink, color in zip(lines, ((5, 3, 35, 10), (3, 10, 40, 23)), ("white", "orange"), strict=True):
        image = Image.new("RGBA", (line["w"], line["h"]))
        image.paste(color, ink)
        image.save(tmp_path / line["file"])
    layout = {"size": [100, 100], "artwork": {"x": 0, "y": 0},
              "texts": [{"role": "headline", "lines": lines}]}
    foreground = np.zeros((100, 100), bool)
    foreground[25, 20] = blocked
    before = scene_ad.rendered_text_bounds(layout, tmp_path)
    scene_ad.tighten_aligned_stack(lines, foreground, layout, tmp_path)
    after = scene_ad.rendered_text_bounds(layout, tmp_path)
    if blocked:
        assert after == before
    else:
        assert after["headline_line_2"][1] - after["headline_line_1"][3] == 3
        assert after["headline_line_1"][3] - after["headline_line_1"][1] == 7
        assert after["headline_line_2"][3] - after["headline_line_2"][1] == 13
        scene_ad.tighten_aligned_stack(lines, foreground, layout, tmp_path)
        assert scene_ad.rendered_text_bounds(layout, tmp_path) == after


def test_balanced_composition_keeps_line_edits_and_moderates_motion_toward_center():
    original = {"boxes": {"headline": [8, 10, 80, 16], "price": [8, 82, 20, 5]}}
    revised = {"boxes": {"headline": [8, 20, 80, 16], "price": [8, 74, 20, 5]},
               "styles": {"headline": {"lines": [{"box": [8, 20, 80, 6], "size": 8},
                                                 {"box": [8, 28, 80, 8], "size": 11}]}}}
    before, after = {"top": 10, "bottom": 13}, {"top": 20, "bottom": 21}
    blended = scene_ad.balanced_text_positions(original, revised, (100, 180), before, after)
    assert blended["boxes"] == {"headline": [8, 15, 80, 16], "price": [8, 78, 20, 5]}
    assert blended["styles"]["headline"]["lines"] == [{"box": [8, 15, 80, 6], "size": 8},
                                                       {"box": [8, 23, 80, 8], "size": 11}]
    assert scene_ad.balanced_text_positions(original, revised, (100, 180), before, after | {"bottom": 12}) is None


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
