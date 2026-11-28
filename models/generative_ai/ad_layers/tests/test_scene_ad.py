import json

import numpy as np
from PIL import Image
import pytest

from models.generative_ai.ad_layers import format_layout, scene_ad
from models.generative_ai.ad_layers.layers import compose
from models.generative_ai.ad_layers.text_render import page_html
from models.generative_ai.photo_retouch import product_mask


@pytest.mark.parametrize(("box", "error"), [
    ([10, 43, 18, 20], None),
    ([0, 43, 18, 20], None),
    ([10, 43, 2, 20], "가장자리"),
    ([55, 45, 18, 20], "실루엣"),
    ([11, 10, 18, 20], "글자"),
])
def test_companion_placement_reserves_visible_space_in_artwork(box, error):
    artwork = {"x": 10, "y": 12, "w": 100, "h": 120}
    subject = Image.new("RGBA", (100, 120))
    subject.paste("white", (53, 46, 85, 93))
    lettering = {"headline": [9, 10, 24, 25], "price": [9, 75, 24, 90]}
    companion = {"object": "small matchbox", "cue": "The full lid and side are visible", "box": box}
    if error:
        with pytest.raises(ValueError, match=error):
            scene_ad.planned_companion({"companion": companion}, artwork, subject, (130, 160), lettering)
    else:
        positioned = scene_ad.planned_companion({"companion": companion}, artwork, subject, (130, 160), lettering)
        assert positioned == companion | {"box": [max(4, box[0]), *box[1:]]}


def test_companion_insertion_moves_away_from_actual_photographed_product(monkeypatch, tmp_path):
    scene = Image.new("RGB", (100, 120), "tan")
    foreground = np.zeros((120, 100), dtype=bool)
    foreground[35:80, 53:88] = True
    companion = {"object": "small matchbox", "cue": "The full lid and side are visible", "box": [55, 45, 18, 20]}
    layout = {"artwork": {"x": 0, "y": 0, "w": 100, "h": 120}, "size": (100, 120)}
    placements = iter([[55, 45, 18, 20], [10, 43, 18, 20]])
    monkeypatch.setattr(scene_ad, "rendered_text_bounds", lambda *_: {})

    class Agent:
        def ask(self, step, images, prompt, *, model):
            box = next(placements)
            assert step == "ad_companion_insertion" and len(images) == 2 and images[0].shape == (120, 100, 3)
            assert not np.array_equal(images[0], images[1])
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            assert data["insertion_feedback"] == "소품이 큽니다"
            assert data["product_bounds"] == [53, 29, 88, 67]
            if box != companion["box"]:
                assert "실루엣" in data["layout_error"] and data["previous_box"] == companion["box"]
            return json.dumps({"box": box, "instruction": "Rest it on the existing surface.",
                               "no_nearby_reason": "The nearby right crops would be past the edge of the surface."})

    bounds, instruction, placed = scene_ad.plan_companion_insertion(
        Agent(), scene, companion, foreground, layout, tmp_path, "소품이 큽니다", rejected=Image.new("RGB", scene.size))
    assert bounds == (10, 52, 28, 76) and placed["box"] == [10, 43, 18, 20]
    assert instruction == "Rest it on the existing surface. Required visible detail: The full lid and side are visible"


def test_companion_insertion_offers_nearby_boxes_that_clear_the_product_and_type(monkeypatch, tmp_path):
    scene = Image.new("RGB", (100, 120), "tan")
    foreground = np.zeros((120, 100), dtype=bool)
    foreground[32:76, 37:74] = True
    companion = {"object": "open book", "cue": "Visible pages", "box": [60, 76, 18, 20]}
    artwork = {"x": 10, "y": 20, "w": 100, "h": 120}
    layout = {"artwork": artwork, "size": (120, 150)}
    lettering = {"price": [8, 48, 34, 66]}
    pairing_intent = {"owner_idea": "책과 함께 책상 위에 배치", "headline": "책상 위의 고요",
                      "photographic_focus": "Warm light on the right of the product."}
    monkeypatch.setattr(scene_ad, "rendered_text_bounds", lambda *_: lettering)

    class Agent:
        def ask(self, step, images, prompt, *, model):
            assert step == "ad_companion_insertion"
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            assert data["pairing_intent"] == pairing_intent
            raw = Image.new("RGBA", scene.size)
            raw.putalpha(Image.fromarray(foreground).convert("L"))
            for box in data["nearby_boxes"].values():
                assert scene_ad.planned_companion({"companion": companion | {"box": box}}, artwork, raw,
                                                  layout["size"], lettering)["box"] == box
            box = data["nearby_boxes"]["beside_right"]
            assert 74 < box[0] <= 77 and box[1] < companion["box"][1]
            return json.dumps({"slot": "beside_right", "instruction": "Place it on the table."})

    bounds, _, picked = scene_ad.plan_companion_insertion(Agent(), scene, companion, foreground, layout, tmp_path, None,
                                                        pairing_intent=pairing_intent)
    assert bounds[:2] == (picked["box"][0], round(picked["box"][1] * 1.2))

    foreground[:92, :] = True
    narrow_surface = scene_ad.nearby_companion_boxes(foreground, layout, {}, [0, 0, 100, 77], [22, 23])
    assert narrow_surface and all(box[2:] == [31, 16] for box in narrow_surface.values())


def test_companion_crop_must_fit_the_product_relative_size_and_shadow(monkeypatch, tmp_path):
    scene = Image.new("RGB", (100, 100), "tan")
    foreground = np.zeros((100, 100), dtype=bool)
    foreground[20:60, 20:40] = True
    layout = {"artwork": {"x": 0, "y": 0, "w": 100, "h": 100}, "size": (100, 100)}
    companion = {"object": "a serving bowl", "cue": "A complete visible rim", "box": [50, 55, 30, 30]}
    scale = {"size": [34, 21], "reference": "The bowl is wider than the pictured product beside it."}
    choices = iter([[50, 55, 30, 30], [45, 40, 50, 35]])
    monkeypatch.setattr(scene_ad, "rendered_text_bounds", lambda *_: {})

    class Agent:
        def ask(self, step, images, prompt, *, model):
            box = next(choices)
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            assert data["target_size"] == scale
            assert all(candidate[2] >= 43 and candidate[3] >= 27 for candidate in data["nearby_boxes"].values())
            if box != companion["box"]:
                assert "그림자 여유" in data["layout_error"]
            return json.dumps({"slot": "custom", "box": box, "instruction": "Place the bowl on the table.",
                               "no_nearby_reason": "The suggested positions are off the usable support surface."})

    bounds, _, _ = scene_ad.plan_companion_insertion(Agent(), scene, companion, foreground, layout, tmp_path, None,
                                                   scale=scale)
    assert bounds == (45, 40, 95, 75)


def test_companion_size_uses_the_visible_anchor_width():
    scene = Image.new("RGB", (200, 100), "tan")
    foreground = np.zeros((100, 200), dtype=bool)
    foreground[25:60, 24:54] = True
    companion = {"object": "open book", "cue": "Facing pages"}

    class Agent:
        def ask(self, step, images, prompt, *, model):
            assert step == "ad_companion_scale"
            return '{"anchor_box":[10,20,20,45],"projected_ratio":[2,0.6],"reference":"ceramic vessel","size":[2,2]}'

    assert scene_ad.plan_companion_scale(Agent(), scene, foreground, companion, {}) == {
        "size": [30, 18], "reference": "ceramic vessel", "anchor_box": [10, 20, 20, 45]}
    with pytest.raises(ValueError, match="실제 상품 영역"):
        scene_ad.plan_companion_scale(Agent(), scene, np.fliplr(foreground), companion, {})


@pytest.mark.parametrize("finding, target, expected", [
    ("natural", None, None), ("wrong_surface", None, None),
    ("too_small", [42, 20], [42, 20]), ("too_large", [24, 12], [24, 12]),
    ("too_small", [74, 40], [45, 24]),
])
def test_companion_geometry_revises_scale_from_the_rendered_photo(finding, target, expected):
    scene = Image.new("RGB", (100, 100), "tan")
    scene.paste("red", (20, 35, 45, 60))
    scene.paste("blue", (45, 65, 82, 80))
    foreground = np.zeros((100, 100), dtype=bool)
    foreground[35:60, 20:45] = True
    scale = {"size": [32, 16], "anchor_box": [20, 35, 25, 25], "reference": "ceramic jar"}
    companion = {"object": "open book", "cue": "Both pages"}

    class Agent:
        def ask(self, step, images, request, *, model):
            assert step == "ad_companion_geometry"
            assert [image.shape for image in images] == [(100, 100, 3), (25, 50, 3)]
            assert "32" not in request
            return json.dumps({"finding": finding, "observation": "앞쪽 탁자에 놓인 소품을 비교했습니다.",
                               "reference": "ceramic jar", "anchor_box": scale["anchor_box"], "target_size": target})

    accepted, _, revision = scene_ad.review_companion_geometry(
        Agent(), scene, foreground, (40, 60, 90, 85), companion, {}, scale)
    assert accepted == (finding == "natural")
    assert revision == (scale | {"size": expected} if expected else scale)


def test_companion_reframe_reserves_space_without_shrinking_only_the_companion(tmp_path, monkeypatch):
    scene = Image.new("RGB", (1024, 1024), "#A38B71")
    foreground = np.zeros((1024, 1024), dtype=bool)
    foreground[390:820, 93:1010] = True
    lettering = {"headline": [8, 13, 42, 33], "product_name": [8, 90, 23, 96], "price": [25, 90, 38, 96]}
    layout = {"artwork": {"x": 0, "y": 0, "w": 1024, "h": 1024}, "size": scene.size}
    companion = {"object": "open book", "cue": "Facing pages", "box": [68, 67, 23, 24]}
    monkeypatch.setattr(scene_ad, "rendered_text_bounds", lambda *_: lettering)

    subject, reserve = scene_ad.companion_reframe(scene, foreground, companion, {"size": [42, 22]}, layout, tmp_path)

    sx, sy, sw, sh = scene_ad.pixel_box(subject, scene.size)
    placed_subject = Image.new("RGBA", scene.size)
    placed_subject.paste("white", (sx, sy, sx + sw, sy + sh))
    assert scene_ad.planned_companion({"companion": companion | {"box": reserve}}, layout["artwork"],
                                      placed_subject, scene.size, lettering)["box"] == reserve
    zoom = min(sw / 917, sh / 430)
    assert 0.69 <= zoom <= 1.02
    assert reserve[2] * 0.8 >= 42 * zoom and reserve[3] * 0.8 >= 22 * zoom

    class Agent:
        def ask(self, step, images, prompt, *, model):
            assert step == "ad_companion_insertion"
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            assert data["nearby_boxes"]["reserved"] == reserve
            return '{"slot":"reserved","instruction":"Rest the open book on the table."}'

    mask = np.asarray(placed_subject.getchannel("A")) > 0
    scale = scene_ad.scale_companion_for_reframe({"size": [42, 22]}, foreground, mask)
    assert scale["size"] == [round(42 * zoom), round(22 * zoom)]
    with pytest.raises(ValueError, match="비율이나 배열"):
        scene_ad.scale_companion_for_reframe(scale, foreground, foreground[:, ::2])
    _, _, plan = scene_ad.plan_companion_insertion(Agent(), scene, companion | {"box": reserve}, mask,
                                                   layout, tmp_path, None, scale=scale)
    assert plan["box"] == reserve


def test_companion_location_guide_compares_the_nearest_product_detail_with_the_reserved_crop(tmp_path, monkeypatch):
    scene = Image.new("RGB", (200, 200), "tan")
    foreground = np.zeros((200, 200), dtype=bool)
    foreground[64:112, 32:62] = True
    foreground[65:110, 124:174] = True
    layout = {"artwork": {"x": 0, "y": 0, "w": 200, "h": 200}, "size": scene.size}
    companion = {"object": "a book", "cue": "two facing pages", "box": [67, 71, 25, 22]}
    scale = {"size": [14, 11], "reference": "the left vessel", "anchor_box": [16, 32, 15, 24]}
    monkeypatch.setattr(scene_ad, "rendered_text_bounds", lambda *_: {})

    class Agent:
        def ask(self, step, images, prompt, *, model):
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            assert step == "ad_companion_insertion" and data["has_placement_guide"] and len(images) == 2
            assert [67, 71, 25, 22] == data["nearby_boxes"]["reserved"]
            box = data["nearby_boxes"]["in_front_left"]
            assert box[0] + box[2] / 2 < 32 and box[1] < 71
            guide = np.rint(images[1] * 255).astype(np.uint8)
            assert np.any(np.all(guide == [86, 232, 255], axis=-1))
            assert np.any(np.all(guide == [249, 154, 255], axis=-1))
            return '{"slot":"in_front_left","instruction":"Set the book on the same table."}'

    _, _, placed = scene_ad.plan_companion_insertion(Agent(), scene, companion, foreground, layout, tmp_path, None,
                                                     scale=scale)
    assert placed["box"][0] < 32

    recropped = np.zeros_like(foreground)
    recropped[30:150, 20:180] = True
    before = np.zeros_like(foreground)
    before[20:140, 30:190] = True
    assert scene_ad.scale_companion_for_reframe(scale, before, recropped)["anchor_box"] == [11, 37, 15, 24]


@pytest.mark.parametrize("decision", ["same_intent", "required_pairing", "unsupported", "intent_lost", "retry", "repeated_product"])
def test_draft_copy_revision_requires_visual_confirmation(decision):
    photo = np.zeros((12, 12, 3))
    candidate = np.ones((12, 12, 3))
    idea = "조명으로 유리의 질감을 강조"
    brief = {"headline": "창가의 한 방울", "message": "일상적인 사용", "scene": "Window light with a drop."}
    draft = {"headline": "빛을 머금은 유리", "message": "유리에 빛이 비칩니다", "evidence": "병의 유리 테두리에 반사가 보입니다"}
    finding = {"observed_scene": "유리 테두리가 반짝입니다", "message_observation": "방울이 보이지 않습니다"}
    stages = []

    class Agent:
        def ask(self, stage, images, prompt, *, model):
            stages.append(stage)
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            if stage == "ad_draft_photo_copy":
                assert data["owner_idea"] == idea and data["draft"] == {key: brief[key] for key in ("headline", "message")}
                assert images[0] is photo and images[1] is candidate
                if decision == "repeated_product":
                    if len(stages) == 1:
                        return json.dumps({"revision": draft | {"headline": "빛을 머금은 향수"}})
                    assert "전체 상품명" in data["visual_issue"]
                    return json.dumps({"revision": draft | {"first_line": "빛을 머금은"}})
                if len(stages) > 1:
                    assert data["visual_issue"] == "반사가 보입니다" and data["previous_revision"] == draft
                return json.dumps({"revision": None if decision == "required_pairing" else draft | {"first_line": "빛을 머금은"}})
            assert stage == "ad_draft_photo_copy_check" and images[0] is candidate
            assert data["headline"] == draft["headline"]
            assert data["owner_idea"] == idea
            supported = decision != "unsupported" and (decision != "retry" or len(stages) > 2)
            return json.dumps({"message_supported": supported,
                               "intent_preserved": decision != "intent_lost", "message_observation": "반사가 보입니다"})

    response = scene_ad.reconcile_draft_headline(Agent(), {"product_name": "향수"}, idea, brief, photo, candidate, finding)
    pair = ["ad_draft_photo_copy", "ad_draft_photo_copy_check"]
    expected = {"required_pairing": pair[:1], "repeated_product": pair[:1] + pair,
                "retry": pair * 2, "unsupported": pair * 2}
    assert stages == expected.get(decision, pair)
    assert response == ((draft | {"scene": finding["observed_scene"]}, "반사가 보입니다", 2)
                        if decision in ("same_intent", "retry", "repeated_product") else None)


def test_decimal_photo_region_protects_visible_object():
    class Agent:
        def ask(self, step, images, prompt, *, model):
            assert step == "ad_photo_regions"
            return '{"regions":[{"part":"책","points":[[46.6,78.8],[63.8,80.5],[63.7,89.8],[43.6,87.6]]}]}'

    photo = Image.new("RGB", (100, 100), "tan")
    regions = scene_ad.find_photo_regions(Agent(), photo, "An open book beside the product.")
    protected = scene_ad.photo_region_mask(regions, photo.size)
    assert protected[84, 53] and not protected[77, 55] and not protected[84, 40]


@pytest.mark.parametrize("crop_only", [False, True])
def test_scene_patch_addition_excludes_product_from_context(crop_only):
    scene = Image.new("RGB", (100, 100), "tan")
    scene.paste("red", (0, 0, 30, 40))
    foreground = np.zeros((100, 100), dtype=bool)
    if not crop_only:
        foreground[:40, :30] = True

    class Model:
        def edit(self, context, prompt, seed, steps, *, size):
            assert size == context.size == (576, 461)
            assert context.getcolors() == [(576 * 461, (210, 180, 140))]
            assert "Add no other items, repeated products, new surface edges or borders" in prompt
            return Image.new("RGB", size, "blue")

    repaired = scene_ad.repair_scene_patch(Model(), scene, (0, 40, 50, 80), "Keep the surface continuous.",
                                           {"seed": 1, "steps": 25}, secondary_object="a small lamp",
                                           foreground=foreground,
                                           companion_mask=(lambda patch: np.zeros(patch.size[::-1], dtype=bool))
                                           if crop_only else None)
    assert repaired.crop((0, 0, 100, 40)).tobytes() == scene.crop((0, 0, 100, 40)).tobytes()
    assert repaired.getpixel((25, 60)) == ((210, 180, 140) if crop_only else (0, 0, 255))


def test_companion_patch_enlarges_the_silhouette_without_using_pixels_outside_the_crop():
    patch = Image.new("RGB", (200, 180), "tan")
    patch.paste("white", (90, 25, 160, 65))
    mask = np.array(patch.getchannel("R")) == 255
    fitted, moved, (left, top, right, bottom) = scene_ad.fit_companion_patch(patch, mask)
    subject = fitted.getchannel("R").point(lambda value: 255 if value == 255 else 0).getbbox()
    assert 0 <= left < right <= patch.width and 0 < top < bottom <= patch.height
    assert np.asarray(fitted.getchannel("R")).min() > 0
    assert 118 <= subject[2] - subject[0] <= 126
    assert np.all(moved[np.asarray(fitted.getchannel("R")) == 255])
    assert abs(left + (subject[0] + subject[2]) / 2 - patch.width / 2) <= 1
    mask[0, 110] = True
    assert scene_ad.fit_companion_patch(patch, mask) is None


def test_companion_surface_matches_the_original_light_and_keeps_the_contact_shadow():
    surface = Image.new("RGB", (120, 100), (72, 60, 45))
    patch = Image.new("RGB", surface.size, (120, 100, 75))
    patch.paste((67, 55, 42), (40, 60, 80, 66))
    patch.paste("#9c20da", (80, 40, 90, 61))
    patch.paste((180, 160, 136), (40, 38, 80, 60))
    mask = np.zeros((100, 120), dtype=bool)
    mask[38:60, 40:80] = True
    balanced = scene_ad.match_companion_surface(patch, surface, mask)
    assert balanced.getpixel((10, 50)) == surface.getpixel((10, 50))
    assert balanced.getpixel((60, 46))[0] > 100
    assert balanced.getpixel((60, 62))[0] < 0.8 * surface.getpixel((60, 62))[0]
    assert balanced.getpixel((86, 50)) == surface.getpixel((86, 50))


def test_companion_surface_keeps_an_attached_cast_shadow_but_not_a_detached_mark():
    surface = Image.new("RGB", (180, 120), (138, 110, 82))
    patch = surface.copy()
    patch.paste((98, 78, 58), (26, 70, 111, 81))
    patch.paste((98, 78, 58), (20, 41, 43, 48))
    patch.paste((210, 200, 182), (94, 45, 133, 75))
    mask = np.zeros((120, 180), dtype=bool)
    mask[45:75, 94:133] = True

    balanced = scene_ad.match_companion_surface(patch, surface, mask)

    assert balanced.getpixel((62, 76))[0] < 0.85 * surface.getpixel((62, 76))[0]
    assert balanced.getpixel((30, 44)) == surface.getpixel((30, 44))


def test_scene_patch_rejects_a_tiny_companion_even_when_its_whole_outline_is_visible():
    scene = Image.new("RGB", (160, 160), "tan")

    class Model:
        def edit(self, context, prompt, seed, steps, *, size):
            assert "about 44 percent of the width and 38 percent of the height" in prompt
            return context

    def silhouette(patch):
        mask = np.zeros(patch.size[::-1], dtype=bool)
        mask[36:44, 36:44] = True
        return mask

    with pytest.raises(ValueError, match="상품에 맞춰 정한 크기"):
        scene_ad.repair_scene_patch(Model(), scene, (40, 40, 120, 120), "Rest it on the existing surface.",
                                   {"seed": 1, "steps": 25}, secondary_object="a small lamp",
                                   companion_mask=silhouette, target_size=(35, 30))


@pytest.mark.parametrize("choice", ["cleaned", "original"])
def test_photo_cleanup_preserves_action_support_and_compares_actual_repair(monkeypatch, choice):
    from models.generative_ai.photo_retouch import cleanup

    scene = Image.new("RGB", (100, 100), "#957a46")
    foreground = np.zeros((100, 100), dtype=bool)
    foreground[40:60, 40:60] = True
    regions = [{"part": "잡고 있는 손", "points": [[65, 5], [85, 5], [85, 20], [65, 20]]}]
    distractions = [{"box": [70, 9, 5, 5], "observation": "손에 닿은 모양"},
                    {"box": [0, 30, 6, 10], "observation": "왼쪽에 잘린 장식"}]

    def remove(model, original, repairs, subject, rule):
        assert subject[10, 75] and subject[45, 45]
        assert repairs == [{"box": [0, 300, 60, 400], "label": "왼쪽에 잘린 장식"}]
        cleaned = original.copy()
        cleaned[30:40, :6] = 0.5
        return cleaned, [repairs[0]["label"]]

    class Agent:
        def ask(self, step, images, request, *, model):
            assert step == "ad_photo_cleanup_compare" and len(images) == 4
            assert not np.array_equal(images[0], images[1]) and not np.array_equal(images[2], images[3])
            return json.dumps({"choice": choice, "reason": "장면을 실제로 보았습니다."})

    monkeypatch.setattr(cleanup, "load_inpaint_model", object)
    monkeypatch.setattr(cleanup, "remove_distractions", remove)
    photo, result = scene_ad.clean_photo_distractions(Agent(), scene, foreground, regions, distractions, "제품 쓰임")
    if choice == "cleaned":
        assert result["removed"] == ["왼쪽에 잘린 장식"]
        assert photo.getpixel((2, 35)) != scene.getpixel((2, 35))
    else:
        assert result is None and photo is scene
    assert photo.getpixel((75, 10)) == scene.getpixel((75, 10))


def test_continuous_photo_exposes_the_scene_beneath_a_full_offer_panel():
    design = {"edge": "blend", "crop": "preserve", "artwork_box": [0, 0, 65, 100],
              "subject_box": [10, 15, 80, 70], "accent": {"box": [65, 20, 35, 60], "shape": "rectangle"}}
    ink = {"product_name": [70, 30, 90, 40], "price": [70, 50, 85, 60]}

    alternative = scene_ad.continuous_offer_scene(design, ink)

    assert alternative["subject_box"] == [6, 15, 52, 70] and alternative["accent"] is None
    assert scene_ad.continuous_offer_scene(design | {"edge": "framed"}, ink) is None
    assert scene_ad.continuous_offer_scene(design | {"crop": "intentional"}, ink) is None


@pytest.mark.parametrize("phrase,needs_revision", [("옆에서 번지는 빛", True), ("차분한 아침", False), (None, False)])
def test_headline_priority_compares_the_campaign_message_to_rendered_lines(phrase, needs_revision):
    lines = [{"text": "옆에서 번지는 빛", "role": "headline_line_1", "max_font_size": 80},
             {"text": "차분한 아침", "role": "headline_line_2", "max_font_size": 70}]
    layout = {"texts": [{"role": "headline", "lines": lines}, {"role": "product_name", "text": "테이블 램프"}]}
    ink = {"headline_line_1": [5, 5, 40, 10], "headline_line_2": [5, 14, 52, 22]}
    brief = {"message": "램프가 은은하게 비추는 아침 공간"}

    class Agent:
        def ask(self, step, images, request, *, model):
            assert step == "ad_headline_priority" and len(images) == 1
            inputs = json.loads(request.split("Input JSON follows:\n", 1)[1])
            assert inputs == {"message": brief["message"], "product_name": "테이블 램프",
                              "headline_lines": [line["text"] for line in lines]}
            return json.dumps({"phrase": phrase, "reason": "광고의 중심 표현입니다."})

    priority = scene_ad.find_headline_priority(Agent(), layout, ink, brief, Image.new("RGB", (60, 60), "orange"))
    assert bool(priority) is needs_revision
    if priority:
        assert priority["phrase"] == phrase
        balanced_ink = ink | {"headline_line_1": [5, 5, 48, 13]}
        assert not scene_ad.weaker_headline_focus(layout, balanced_ink, phrase)


def test_headline_scale_transfer_keeps_reading_order_and_title_extent():
    lines = [{"box": [5, 5, 40, 7], "size": 5}, {"box": [5, 14, 43, 10], "size": 8}]
    design = {"styles": {"headline": {"size": 8, "lines": lines}, "price": {"size": 4}}}
    layout = {"texts": [{"role": "headline", "lines": [{"text": "조명을 켜면"}, {"text": "빛이 번져요"}]}]}

    styles = scene_ad.transfer_headline_scale(design, layout, "조명을")

    assert styles["headline"]["lines"] == [
        {"box": [5, 5, 40, 10], "size": 8}, {"box": [5, 17, 43, 7], "size": 5},
    ]
    assert styles["price"] == design["styles"]["price"]
    assert lines[0]["size"] == 5


@pytest.mark.parametrize("fragments,identity", [(True, True), (False, False), (False, None)])
def test_source_framing_keeps_whole_when_detail_loses_the_subject(monkeypatch, fragments, identity):
    crop = [25, 25, 50, 50]
    product = Image.new("RGBA", (80, 60), "orange")
    review = {"choice": "detail", "reason": "원단 무늬는 보이지만 제품의 형태를 확인할 수 없습니다."}
    if identity is not None:
        review["detail_preserves_identity"] = identity

    def ask_stage(ai, step, task, inputs, images):
        if step == "ad_source_framing":
            return {"source_crop": crop, "reason": "원단 무늬를 확대합니다."}
        if step == "ad_source_detail_observation":
            assert inputs == {} and len(images) == 1
            return {"observation": review["reason"], "competing_fragments": fragments}
        assert step == "ad_source_framing_check" and not fragments
        assert "proposal" not in inputs
        assert inputs["detail_observation"] == review["reason"]
        return review

    monkeypatch.setattr(scene_ad, "ask_stage", ask_stage)
    original = np.zeros((60, 80, 3))
    selected = scene_ad.choose_source_framing(None, product, {}, [original])
    assert selected["source_crop"] is None


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
    reject_identity = False
    bad_background_shots = 0
    omit_background_verdict = False
    cleaner_scene = "A seamless softly lit gray backdrop continues behind the original glass bottle."
    repair_writing = False
    legacy_photo = False
    repair_message = False
    repair_scene_message = False
    scene_cue = "A small lamp sits fully in frame beside the bottle."
    scene_correction = "Keep the bottle in place by the morning window with a visible lamp beside it."
    action_cue = "Soft sunlight falls across an in-use product beside a morning window."
    support, target = "Fingers grasp the product cap in frame.", "The tool reaches a clean surface apart from the bottle."
    corrected_cue = f"{action_cue} Visible support: {support} Visible destination: {target}"
    semantic_correction = "Show the product in use near the window so the image carries the morning message."
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
            if repair_scene_message and "ad_photo_cue_check" in calls:
                assert size == reference.size and max(size) == 576
                assert reference.getpixel((0, 0)) == (0, 0, 255)
                assert scene_correction in prompt and scene_cue in prompt and "Visible destination" not in prompt
                assert "silhouette inside (29, 5, 71, 77) percent" in prompt and "Intended object: a small lamp" in prompt
                calls.append("qwen-retouch")
                return Image.new("RGB", size, "magenta")
            assert size == (512, 1024)
            if bad_background_shots and "ad_photo_cue_check" in calls:
                assert prompt.startswith("Priority reshoot correction.")
                assert cleaner_scene in prompt and "Use only the named scene elements" in prompt
                calls.append("qwen")
                return Image.new("RGB", size, "blue")
            if repair_message and "ad_photo_cue_check" in calls:
                assert reference.tobytes() == photo.tobytes()
                assert corrected_cue in prompt and semantic_correction in prompt and cue not in prompt
                calls.append("qwen")
                return Image.new("RGB", size, "blue")
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
            if reject_identity and "ad_photo_cue_check" in calls:
                assert "Restore the original amber glass bottle." in prompt
            assert (cue in prompt) is not legacy_photo
            calls.append("qwen")
            return Image.new("RGB", size, (160, 40, 255) if is_revision else (0, 0, 255))

    class Interpreter:
        def ask(self, step, images, prompt, *, model):
            assert model == "gpt-6-luna"
            if step == "ad_photo_observation":
                assert len(images) == 1 and json.loads(prompt.split("Input JSON follows:\n", 1)[1]) == {}
                assert "부드러운 아침" not in prompt
                return '{"observation":"유리 가장자리에 호박색 빛이 있습니다"}'
            if step == "ad_photo_message":
                data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
                assert len(images) == 1 and images[0].shape == (1024, 512, 3)
                assert data.pop("product_facts") == fields
                assert isinstance(data.pop("owner_idea"), str) and data.pop("photo_cue")
                assert data == {"headline": "부드러운 아침", "message": "아침에 쓰는 세럼",
                                "observation": "유리 가장자리에 호박색 빛이 있습니다"}
                if repair_message and "ad_photo_cue_check" not in calls:
                    return json.dumps({"message_supported": False, "message_observation": "아침 장면이 보이지 않습니다",
                                       "correction": semantic_correction, "corrected_photo_cue": action_cue,
                                       "visible_support": support, "action_target": target,
                                       "reposition": True})
                if repair_scene_message and "ad_photo_cue_check" not in calls:
                    return json.dumps({"message_supported": False, "message_observation": "램프가 보이지 않습니다.",
                                       "correction": scene_correction, "corrected_photo_cue": scene_cue,
                                       "repair_kind": "scene", "reposition": False, "repair_box": [50, 0, 30, 30],
                                       "secondary_object": "a small lamp"})
                return '{"message_supported":true,"message_observation":"사진에 아침빛이 보입니다"}'
            calls.append(step)
            if step == "ad_photo_patch_check":
                data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
                assert data["secondary_object"] == "a small lamp"
                before, after, close = images
                assert np.array_equal(before[307:], after[307:])
                assert np.array_equal(before[:, :256], after[:, :256])
                assert np.array_equal(before[:, 410:], after[:, 410:])
                assert np.array_equal(close[0, 60], [1, 0, 1])
                return '{"accepted":true,"observation":"새 조명이 주변과 자연스럽게 이어집니다"}'
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
            if step == "ad_source_detail_observation":
                assert len(images) == 1
                return '{"observation":"앰버색 유리 가장자리가 보입니다","competing_fragments":false}'
            if step == "ad_source_framing_check":
                assert expect_correction and images[0].shape == images[1].shape == (512, 512, 3)
                assert not np.array_equal(images[0], images[1])
                return ('{"choice":"detail","detail_preserves_identity":true,'
                        '"reason":"병의 유리 가장자리에서 앰버빛이 확대되어 보입니다"}')
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
                assert data["headline"] == "부드러운 아침"
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
                data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
                has_corrected_message = repair_message and calls.count(step) == 2
                expected_cue = "유리 가장자리에 호박색 빛이 있습니다" if has_corrected_message else cue
                if repair_scene_message and calls.count(step) == 2:
                    expected_cue = scene_cue
                assert data["photo_cue"] == (None if legacy_photo else expected_cue)
                assert images[1].shape == (1024, 512, 3)
                assert data["has_approved_detail"] is expect_correction and len(images) == (7 if expect_correction else 6)
                assert [image.shape for image in images[-4:]] == [(640, 320, 3)] * 4
                identity = {"identity_matches": not reject_identity,
                            "identity_observation": "병이 검정색입니다" if reject_identity else "같은 앰버색 병입니다",
                            "stray_writing": None}
                if bad_background_shots and not omit_background_verdict:
                    identity["blocking_distraction"] = None
                if calls.count("qwen") <= bad_background_shots:
                    return json.dumps(identity | {"visible": True, "observation": "상품은 같고 뒤에 상자가 있습니다.",
                                                  "blocking_distraction": {"observation": "상품 뒤에 큰 상자가 있습니다.",
                                                                           "scene": cleaner_scene}})
                if legacy_photo and calls.count(step) == 1:
                    return '{"identity_matches":true,"visible":true,"identity_observation":"같은 병입니다","observation":"병이 보입니다"}'
                if legacy_photo:
                    assert "The previous answer omitted stray_writing." in prompt
                if repair_writing and calls.count(step) == 1:
                    return json.dumps(identity | {"visible": True, "observation": "빛이 보입니다",
                                                  "stray_writing": {"box": [70, 10, 5, 2], "observation": "배경에 글자가 있습니다"},
                                                  "correction": "Remove the background lettering.", "reposition": False})
                if repair_writing:
                    assert images[1][1, 1].tolist() == [1, 0, 0]
                if reject_identity:
                    return json.dumps(identity | {"visible": True, "observation": "빛이 보입니다",
                                                  "correction": "Restore the original amber glass bottle.",
                                                  "reposition": False})
                if reject_cue or exercise_cue_retry and calls.count(step) == 1:
                    return json.dumps(identity | {"visible": False, "observation": "병 가장자리에 빛이 없습니다",
                                                  "correction": cue_correction, "reposition": reject_cue})
                return json.dumps(identity | {"visible": True, "observation": "유리 가장자리에 빛이 보입니다",
                                              "correction": None})
            if step == "ad_photo_regions":
                assert len(images) == 1 and "photo_cue" in prompt
                return '{"regions":[]}'
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
            if step == "ad_photo_type_placement":
                assert len(images) >= 2
                return '{"choice":"original","reason":"정리한 제목 배치를 유지합니다."}'
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
            ink.paste("white", (0, 0, 8, 46 if text["role"] == "headline" else 36))
            ink.save(folder / text["file"])

    monkeypatch.setattr(product_mask, "product_mask", mask)
    monkeypatch.setattr(scene_ad, "render_texts", render)
    folder = tmp_path / "result"
    models = {"ai": Interpreter(), "generation": Model(), "mask": "mask"}
    layout = scene_ad.make_scene_ad(photo, fields, "", models, folder, references=[reference])
    expected = ["ad_plan", "ad_direction", "ad_headline", "ad_art_direction", "ad_source_framing", "ad_layout",
                "ad_layout", "render", "ad_composition", "render", "ad_composition_compare", "ad_scene_direction", "qwen",
                "ad_photo_cue_check", "qwen", "ad_photo_cue_check", "ad_photo_regions", "ad_final_layout", "render"]
    assert calls[:len(expected)] == expected
    assert set(calls[len(expected):]) <= {"render", "ad_photo_type_placement"}
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
    assert calls == ["render", "qwen", "ad_photo_cue_check", "ad_photo_regions"]
    assert accepted["design"] == layout["design"] and accepted["final_review_reason"] is None
    calls.clear()
    legacy_photo = True
    legacy = {key: value for key, value in layout.items() if key != "photo_cue"}
    checked = scene_ad.make_scene_ad(photo, fields, "", models, tmp_path / "legacy-checked", proposal=legacy)
    assert calls == ["render", "qwen", "ad_photo_cue_check", "ad_photo_cue_check", "ad_photo_regions"]
    assert checked["photo_cue_checks"][0]["stray_writing"] is None
    legacy_photo = False
    repair_message = True
    calls.clear()
    corrected = scene_ad.make_scene_ad(photo, fields, "", models, tmp_path / "message-corrected", proposal=layout)
    assert calls == ["render", "qwen", "ad_photo_cue_check", "qwen", "ad_photo_cue_check", "ad_photo_regions"]
    assert [check["message_supported"] for check in corrected["photo_cue_checks"]] == [False, True]
    assert corrected["photo_cue"] == "유리 가장자리에 호박색 빛이 있습니다"
    assert corrected["design"]["art_direction"] == semantic_correction
    assert corrected["texts"] == layout["texts"]
    repair_message = False
    repair_scene_message = True
    calls.clear()
    corrected_scene = scene_ad.make_scene_ad(photo, fields, "", models, tmp_path / "scene-corrected", proposal=layout)
    assert calls == ["render", "qwen", "ad_photo_cue_check", "qwen-retouch", "ad_photo_cue_check",
                     "ad_photo_patch_check", "ad_photo_regions"]
    assert [check["message_supported"] for check in corrected_scene["photo_cue_checks"]] == [False, True]
    assert corrected_scene["photo_cue"] == scene_cue and corrected_scene["texts"] == layout["texts"]
    repair_scene_message = False
    reject_cue = True
    calls.clear()
    unapproved = tmp_path / "unapproved"
    with pytest.raises(ValueError, match="사진에 의도한 연출이 보이지 않습니다"):
        scene_ad.make_scene_ad(photo, fields, "", models, unapproved, proposal=layout)
    assert calls == ["render", "qwen", "ad_photo_cue_check", "qwen", "ad_photo_cue_check"]
    assert not unapproved.exists()
    reject_cue = False
    reject_identity = True
    calls.clear()
    wrong_product = tmp_path / "wrong_product"
    with pytest.raises(ValueError, match="촬영한 사진의 상품이 원본과 다릅니다. 병이 검정색입니다"):
        scene_ad.make_scene_ad(photo, fields, "", models, wrong_product, proposal=layout)
    assert calls == ["render", "qwen", "ad_photo_cue_check", "qwen", "ad_photo_cue_check"]
    assert not wrong_product.exists()
    reject_identity = False
    bad_background_shots = 1
    calls.clear()
    clean_background = scene_ad.make_scene_ad(photo, fields, "", models, tmp_path / "clean-background", proposal=layout)
    assert calls == ["render", "qwen", "ad_photo_cue_check", "qwen", "ad_photo_cue_check", "ad_photo_regions"]
    first, final = clean_background["photo_cue_checks"]
    assert first["blocking_distraction"] == "상품 뒤에 큰 상자가 있습니다." and "blocking_distraction" not in final
    assert clean_background["texts"] == layout["texts"]
    assert clean_background["design"]["art_direction"] == cleaner_scene
    bad_background_shots = 2
    calls.clear()
    blocked = tmp_path / "blocked-background"
    with pytest.raises(ValueError, match="상품과 경쟁하는 배경 물체가 남았습니다"):
        scene_ad.make_scene_ad(photo, fields, "", models, blocked, proposal=layout)
    assert calls.count("qwen") == 2 and not blocked.exists()
    bad_background_shots = 1
    omit_background_verdict = True
    calls.clear()
    unchecked = tmp_path / "unchecked-background"
    with pytest.raises(ValueError, match="배경 물체가 정리됐는지 확인하지 못했습니다"):
        scene_ad.make_scene_ad(photo, fields, "", models, unchecked, proposal=layout)
    assert calls.count("qwen") == 2 and calls.count("ad_photo_cue_check") == 3 and not unchecked.exists()
    omit_background_verdict = False
    bad_background_shots = 0
    repair_writing = True
    calls.clear()

    def remove_writing(scene, foreground, mark):
        assert mark["box"] == [70, 10, 5, 2] and foreground.any()
        result = scene.copy()
        result.putpixel((1, 1), (255, 0, 0))
        return result

    monkeypatch.setattr(scene_ad, "remove_photo_writing", remove_writing)
    cleaned = scene_ad.make_scene_ad(photo, fields, "", models, tmp_path / "cleaned", proposal=layout)
    assert calls == ["render", "qwen", "ad_photo_cue_check", "ad_photo_cue_check", "ad_photo_regions"]
    assert [check["stray_writing"] for check in cleaned["photo_cue_checks"]] == ["배경에 글자가 있습니다", None]
    repair_writing = False
    calls.clear()
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    (rejected / "ad.png").write_bytes((folder / "ad.png").read_bytes())
    rejected_layout = layout | {"revision": {"visual_review": "병의 하단이 잘렸습니다"}}
    (rejected / "layout.json").write_text(json.dumps(rejected_layout), encoding="utf-8")
    revision = scene_ad.make_scene_ad(photo, fields, "가벼운 아침 연출", models, tmp_path / "revision", references=[reference],
                                      previous=folder, rejected=rejected)
    expected = ["ad_cycle_critique", "ad_art_direction", "ad_source_framing", "ad_layout", "ad_layout", "render",
                "ad_composition", "render", "ad_composition_compare", "ad_scene_direction", "qwen", "ad_photo_cue_check",
                "ad_photo_regions", "ad_final_layout", "render"]
    assert calls[:len(expected)] == expected
    assert set(calls[len(expected):]) <= {"render", "ad_photo_type_placement"}
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


def test_photo_review_close_views_cover_outer_edges_and_overlap_at_the_center():
    scene = Image.new("RGB", (59, 83))
    positions = [(0, 0), (58, 0), (0, 82), (58, 82)]
    for i, corner in enumerate(positions):
        scene.putpixel(corner, (0, 40 + i, 255))
    scene.putpixel((29, 41), (255, 250, 90))
    for i, (tile, (x, y)) in enumerate(zip(scene_ad.photo_review_details(scene), positions, strict=True)):
        assert tile[-1 if y else 0, -1 if x else 0, 1] == pytest.approx((40 + i) / 255)
        assert np.any(tile[:, :, 0] == 1)


@pytest.mark.parametrize("region_key", ["action_regions", "photo_regions"])
def test_text_keeps_supporting_forms_clear_even_outside_product_mask(tmp_path, region_key):
    foreground = np.zeros((40, 40), dtype=bool)
    foreground[5:12, 4:10] = True
    artwork = {"x": 10, "y": 10}
    regions = [{"part": "받침", "points": [[25, 70], [60, 70], [65, 95], [25, 95]]}]
    layer = {"role": "price", "file": "price.png", "x": 21, "y": 40, "w": 9, "h": 6}
    layout = {"size": [60, 60], "artwork": artwork, "texts": [layer]}
    Image.new("RGBA", (9, 6), "white").save(tmp_path / "price.png")
    assert not scene_ad.check_text_overlap(foreground, layout, tmp_path)
    layout[region_key] = regions
    with pytest.raises(ValueError, match="사용 장면"):
        scene_ad.check_text_overlap(foreground, layout, tmp_path)
    layer["y"] = 30
    assert not scene_ad.check_text_overlap(foreground, layout, tmp_path)


def test_nearby_offer_uses_the_headline_axis_only_for_a_small_move():
    ink = {"headline": [7, 5, 52, 10], "product_name": [4, 16, 20, 21], "price": [4, 23, 13, 27]}
    plan = {"boxes": {"headline": [6, 4, 70, 11], "product_name": [3, 15, 24, 6], "price": [3, 22, 24, 5]},
            "styles": {role: {"align": "left"} for role in ink}}
    revised = scene_ad.align_nearby_offer(plan, ink, (1024, 1024))
    assert revised["boxes"] == plan["boxes"] | {"product_name": [6, 15, 24, 6], "price": [6, 22, 24, 5]}
    far_title = ink | {"headline": [38, 5, 95, 10]}
    assert scene_ad.align_nearby_offer(plan, far_title, (1024, 1024)) is None


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
    lines = [{"box": [10, 10, 60, 12], "size": 6, "text": "따뜻한 빛으로", "break_after": 2},
             {"box": [25, 35, 60, 12], "color": "#ffa040", "break_after": 2}]
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
    lines[1]["break_after"] = 1
    with pytest.raises(ValueError, match="제목 전체의 줄바꿈과 다릅니다"):
        scene_ad.layout_texts(answer, copy, (100, 100), ["noto_sans_kr"])
    del lines[1]["break_after"]
    lines[0]["text"] = "다른 문구"
    with pytest.raises(ValueError, match="확정된 원문"):
        scene_ad.layout_texts(answer, copy, (100, 100), ["noto_sans_kr"])
    del lines[0]["text"]
    lines[1]["box"] = [25, 51, 60, 12]
    with pytest.raises(ValueError, match="제목 전체 영역"):
        scene_ad.layout_texts(answer, copy, (100, 100), ["noto_sans_kr"])


@pytest.mark.parametrize(("product_above", "blocked"), [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("between_boxes", [1, 25])
def test_split_title_closes_the_visible_gap_without_covering_the_product(tmp_path, product_above, blocked, between_boxes):
    lines = [
        {"role": "headline_line_1", "file": "first.png", "x": 10, "y": 10, "w": 50, "h": 20},
        {"role": "headline_line_2", "file": "second.png", "x": 10, "y": 30 + between_boxes, "w": 50, "h": 25},
    ]
    for line, ink, color in zip(lines, ((5, 3, 35, 10), (3, 10, 40, 23)), ("white", "orange"), strict=True):
        image = Image.new("RGBA", (line["w"], line["h"]))
        image.paste(color, ink)
        image.save(tmp_path / line["file"])
    layout = {"size": [100, 100], "artwork": {"x": 0, "y": 0},
              "texts": [{"role": "headline", "lines": lines}]}
    foreground = np.zeros((100, 100), bool)
    foreground[4, 50] = product_above
    foreground[34 + between_boxes, 20] = blocked
    before = scene_ad.rendered_text_bounds(layout, tmp_path)
    scene_ad.tighten_text_groups(foreground, layout, tmp_path, {"headline": {}})
    after = scene_ad.rendered_text_bounds(layout, tmp_path)
    if blocked:
        assert after == before
    else:
        assert after["headline_line_2"][1] - after["headline_line_1"][3] == 3
        assert after["headline_line_1"][3] - after["headline_line_1"][1] == 7
        assert after["headline_line_2"][3] - after["headline_line_2"][1] == 13
        anchor = "headline_line_1" if product_above else "headline_line_2"
        assert after[anchor] == before[anchor]
        scene_ad.tighten_text_groups(foreground, layout, tmp_path, {"headline": {}})
        assert scene_ad.rendered_text_bounds(layout, tmp_path) == after


def test_offer_reflow_keeps_the_position_and_pixels_of_the_finished_title(tmp_path, monkeypatch):
    size = (100, 100)
    fonts = ["noto_sans_kr"]
    copy = {"headline": "따뜻한 오후 차 한 잔", "product_name": "차", "price": "4,000원"}
    style = {"font": fonts[0], "align": "left", "color": "#FFFFFF", "size": 6, "max_lines": 1}
    design = {"boxes": {"headline": [8, 8, 80, 26], "product_name": [8, 48, 10, 8], "price": [8, 58, 15, 7]},
              "styles": {role: style.copy() for role in copy}}
    design["styles"]["headline"].update(max_lines=2, break_after=2,
                                        lines=[{"box": [8, 8, 80, 8]}, {"box": [8, 23, 80, 10]}])
    texts = scene_ad.layout_texts(design, copy, size, fonts)
    title, name, price = texts
    title["lines"][0]["y"] += 8
    for line in title["lines"]:
        png = Image.new("RGBA", (line["w"], line["h"]))
        png.paste("#d7a46a", (0, 1, 25, 6))
        png.save(tmp_path / line["file"])
    for text in (name, price):
        png = Image.new("RGBA", (text["w"], text["h"]))
        png.paste("white", (0, 3, 7, 5))
        png.save(tmp_path / text["file"])
    for filename in ("background.png", "scene.png"):
        Image.new("RGB", size, "#221f1b").save(tmp_path / filename)
    foreground = np.zeros(size, dtype=bool)
    foreground[37:74, 24:80] = True
    Image.fromarray(foreground).save(tmp_path / "foreground.png")
    layout = {"size": list(size), "background": {"file": "background.png"},
              "artwork": {"file": "scene.png", "x": 0, "y": 0, "w": 100, "h": 100}, "texts": texts}
    title_pixels = {line["file"]: (tmp_path / line["file"]).read_bytes() for line in title["lines"]}
    plan = design | {"boxes": design["boxes"] | {"product_name": [8, 81, 10, 8], "price": [20, 81, 20, 8]}}
    def offer_reflow_options(_layout, _design, _occupied, _bounds, ink_heights, target_height):
        assert ink_heights == {"product_name": 2, "price": 2}
        assert target_height == 4
        return {"below": plan}

    monkeypatch.setattr(format_layout, "offer_reflow_options", offer_reflow_options)

    def render_offer(offer, _size, path):
        assert {layer["role"] for layer in offer} == {"product_name", "price"}
        for layer in offer:
            png = Image.new("RGBA", (layer["w"], layer["h"]))
            png.paste("white", (0, 0, 7, 5))
            png.save(path / layer["file"])

    class Agent:
        def ask(self, step, images, request, *, model):
            assert step == "ad_crowded_offer_compare"
            np.testing.assert_array_equal(images[0][:34], images[1][:34])
            return '{"choice":"below","reason":"판매 정보를 함께 읽기 쉽습니다."}'

    monkeypatch.setattr(scene_ad, "render_texts", render_offer)
    result, revised, reason = scene_ad.review_crowded_offer(Agent(), design, layout, foreground, tmp_path, fonts)
    assert result == plan and reason
    assert revised["texts"][0] == title
    assert revised["texts"][2]["y"] == 81
    assert {file: (tmp_path / file).read_bytes() for file in title_pixels} == title_pixels


def test_photo_type_check_uses_visible_size_and_counts_competing_color_treatments():
    design = {"focus": "product", "styles": {"headline": {"max_lines": 2,
                                                          "emphasis": {"color": "#DE97A2"}}},
              "accent": {"color": "#C44235"}}
    ink = {"headline": [8, 10, 58, 20], "headline_line_1": [8, 10, 35, 11],
           "headline_line_2": [8, 13, 58, 20], "product_name": [8, 77, 26, 80], "price": [8, 85, 40, 94]}
    issues = scene_ad.photo_type_issues(design, ink, (1024, 1024))
    assert [issue.split(" ", 1)[0] for issue in issues] == ["제목의", "가격의", "제목의"]
    ink.update(headline_line_1=[8, 10, 35, 13], price=[8, 85, 40, 89])
    assert scene_ad.photo_type_issues(design | {"accent": None}, ink, (1024, 1024)) == []


@pytest.mark.parametrize("outcome", ["joined", "original", "obstructed"])
def test_related_offer_moves_both_layers_without_changing_type_pixels(tmp_path, outcome):
    size = (200, 200)
    fonts = ["noto_sans_kr"]
    copy = {"headline": "부드럽게 펴 바르는", "product_name": "수분 크림", "price": "12,000원"}
    style = {"font": fonts[0], "align": "left", "color": "#FFFFFF", "size": 10, "max_lines": 1}
    design = {"boxes": {"headline": [20, 15, 60, 22], "product_name": [20, 46, 30, 12],
                        "price": [58, 46, 24, 12]}, "styles": {role: style.copy() for role in copy}}
    texts = scene_ad.layout_texts(design, copy, size, fonts)
    for layer, area in zip(texts, ((0, 21, 70, 39), (0, 4, 54, 22), (0, 4, 40, 22)), strict=True):
        image = Image.new("RGBA", (layer["w"], layer["h"]))
        image.paste("white", area)
        image.save(tmp_path / layer["file"])
    for filename in ("background.png", "scene.png"):
        Image.new("RGB", size, "#211B16").save(tmp_path / filename)
    foreground = np.zeros(size, dtype=bool)
    foreground[84, 42] = outcome == "obstructed"
    layout = {"size": list(size), "background": {"file": "background.png"}, "texts": texts,
              "artwork": {"file": "scene.png", "x": 0, "y": 0, "w": 200, "h": 200}}
    pixels = [(tmp_path / layer["file"]).read_bytes() for layer in texts]

    class Agent:
        def ask(self, step, images, request, *, model):
            assert outcome != "obstructed" and step == "ad_related_offer_compare"
            assert len(images) == 2 and not np.array_equal(*images)
            return json.dumps({"choice": outcome, "reason": "상품명과 제목 사이의 간격을 확인했습니다."})

    revised, candidate, reason = scene_ad.review_related_offer(Agent(), design, layout, foreground, tmp_path)
    assert bool(reason) == (outcome == "joined")
    assert revised["styles"] == design["styles"]
    assert candidate["texts"][0] == texts[0]
    assert [layer["y"] - text["y"] for layer, text in zip(candidate["texts"][1:], texts[1:], strict=True)] == (
        [-16, -16] if reason else [0, 0])
    assert [(tmp_path / layer["file"]).read_bytes() for layer in texts] == pixels


@pytest.mark.parametrize("focus,blocked", [("headline", False), ("headline", True), ("price", False)])
def test_headline_scale_keeps_the_original_photo_and_selling_information(tmp_path, monkeypatch, focus, blocked):
    size = (200, 200)
    fonts = ["noto_sans_kr"]
    copy = {"headline": "저녁을 밝히는 빛", "product_name": "향초", "price": "18,000원"}
    style = {"font": fonts[0], "align": "left", "color": "#FFFFFF", "size": 10, "max_lines": 1}
    design = {"focus": focus, "boxes": {"headline": [15, 10, 60, 20], "product_name": [15, 47, 28, 14],
                                        "price": [15, 64, 28, 14]},
              "styles": {role: style.copy() for role in copy}}
    texts = scene_ad.layout_texts(design, copy, size, fonts)

    def draw(layer, path, bounds):
        image = Image.new("RGBA", (layer["w"], layer["h"]))
        image.paste("white", bounds)
        image.save(path / layer["file"])

    for text, bounds in zip(texts, ((1, 10, 54, 25), (1, 5, 41, 23), (1, 5, 51, 23)), strict=True):
        draw(text, tmp_path, bounds)
    for filename in ("background.png", "scene.png"):
        Image.new("RGB", size, "#26201B").save(tmp_path / filename)
    layout = {"size": list(size), "background": {"file": "background.png"}, "texts": texts,
              "artwork": {"file": "scene.png", "x": 0, "y": 0, "w": 200, "h": 200}}
    foreground = np.zeros(size, dtype=bool)
    foreground[40, 100] = blocked
    original = [(tmp_path / name).read_bytes() for name in ("scene.png", "text_product_name.png", "text_price.png")]

    def render(words, size, folder):
        assert len(words) == 1 and words[0]["max_font_size"] > texts[0]["max_font_size"]
        draw(words[0], folder, (1, 6, 80, 30))

    class Agent:
        def ask(self, step, images, request, *, model):
            assert focus == "headline" and not blocked
            assert step == "ad_headline_scale_compare" and len(images) == 2
            return '{"choice":"larger","reason":"제목이 먼저 보입니다."}'

    monkeypatch.setattr(scene_ad, "render_texts", render)
    monkeypatch.setattr(scene_ad, "fitted_text_sizes", lambda *_: {"headline": 20})
    revised, updated, reason = scene_ad.review_headline_scale(Agent(), design, layout, foreground, tmp_path, fonts)
    assert bool(reason) == (focus == "headline" and not blocked)
    assert revised["boxes"] == design["boxes"] and updated["texts"][1:] == texts[1:]
    assert updated["texts"][0]["text"] == copy["headline"]
    assert [(tmp_path / name).read_bytes() for name in ("scene.png", "text_product_name.png", "text_price.png")] == original


def test_headline_scale_reduces_only_the_oversized_line(tmp_path, monkeypatch):
    size = (200, 300)
    fonts = ["noto_sans_kr"]
    copy = {"headline": "저녁을 밝히는 빛", "product_name": "향초", "price": "18,000원"}
    style = {"font": fonts[0], "align": "left", "color": "#FFFFFF", "size": 6, "max_lines": 1}
    title = style | {"max_lines": 2, "break_after": 2, "lines": [
        {"box": [10, 6, 75, 10], "size": 7},
        {"box": [10, 17, 75, 14], "size": 16}]}
    design = {"focus": "headline", "boxes": {"headline": [10, 6, 75, 25], "product_name": [10, 65, 35, 8],
                                            "price": [48, 65, 35, 8]},
              "styles": dict.fromkeys(copy, style) | {"headline": title}}
    texts = scene_ad.layout_texts(design, copy, size, fonts)

    def render(words, size, folder):
        for text in words:
            for line in text.get("lines", (text,)):
                bitmap = Image.new("RGBA", (line["w"], line["h"]))
                bitmap.paste("white", (1, 2, 40, 2 + line["max_font_size"]))
                bitmap.save(folder / line["file"])

    render(texts, size, tmp_path)
    for filename in ("background.png", "scene.png"):
        Image.new("RGB", size, "#26201B").save(tmp_path / filename)
    layout = {"size": list(size), "background": {"file": "background.png"}, "texts": texts,
              "artwork": {"file": "scene.png", "x": 0, "y": 0, "w": 200, "h": 300}}
    foreground = np.zeros((300, 200), dtype=bool)

    class Agent:
        def ask(self, step, images, request, *, model):
            assert step == "ad_headline_scale_compare" and len(images) == 4
            assert images[2].shape == images[3].shape and not np.array_equal(images[2], images[3])
            return '{"choice":"balanced","reason":"두 줄이 자연스럽게 이어집니다."}'

    monkeypatch.setattr(scene_ad, "render_texts", render)
    monkeypatch.setattr(scene_ad, "fitted_text_sizes", lambda rows, _: {
        line["role"]: line["max_font_size"] for line in rows})
    _, updated, reason = scene_ad.review_headline_scale(Agent(), design, layout, foreground, tmp_path, fonts)

    assert reason
    first, second = updated["texts"][0]["lines"]
    assert first["max_font_size"] == 14 and 14 < second["max_font_size"] < 32
    assert updated["texts"][1:] == texts[1:]


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
    barely_changed = {"top": 12, "bottom": 12}
    assert scene_ad.balanced_text_positions(original, revised, (100, 180), before, barely_changed) is None


def test_balanced_composition_moves_offer_accent_when_only_bottom_margin_grows():
    original = {"boxes": {"headline": [7, 6, 45, 16], "product_name": [11, 78, 56, 6], "price": [11, 86, 30, 7]},
                "accent": {"shape": "rectangle", "box": [7, 78, 1, 14]}}
    revised = {"boxes": original["boxes"] | {"product_name": [11, 58, 56, 6], "price": [11, 66, 30, 7]},
               "accent": {"shape": "rectangle", "box": [7, 58, 1, 15]}, "styles": {"headline": {}}}
    before, after = {"top": 8, "bottom": 9}, {"top": 8, "bottom": 29}
    balanced = scene_ad.balanced_text_positions(original, revised, (1024, 1024), before, after)
    assert balanced["boxes"] == original["boxes"] | {"product_name": [11, 68, 56, 6], "price": [11, 76, 30, 7]}
    assert balanced["accent"]["box"] == [7, 68, 1, 15]


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
