"""그래픽 카드와 모델 없이 도는 확인: 보정 값 적용, 업종 규칙, 기본 자동 보정, 충실도 검사."""

import numpy as np

from models.generative_ai.photo_retouch.adjustments import apply, apply_regions, clamp_params, luminance
from models.generative_ai.photo_retouch.auto_params import auto_params
from models.generative_ai.photo_retouch.fidelity import check
from models.generative_ai.photo_retouch.settings import category_rules, load_categories, load_settings

SETTINGS = load_settings()
OPS = SETTINGS["ops"]


def photo(seed=0, size=(64, 96)):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.1, 0.9, size[1], dtype=np.float32)[None, :, None] * np.ones((size[0], 1, 3), np.float32)
    return np.clip(base * np.array([0.9, 0.8, 0.7], np.float32) + rng.normal(0, 0.03, (*size, 3)).astype(np.float32), 0, 1)


def test_empty_params_keep_photo():
    image = photo()
    assert np.abs(apply(image, {}, OPS) - image).max() < 1e-3


def test_each_param_moves_the_right_way():
    image = photo()
    assert luminance(apply(image, {"exposure_ev": 0.5}, OPS)).mean() > luminance(image).mean()
    warm = apply(image, {"temperature": 0.5}, OPS)
    assert warm[..., 0].mean() / warm[..., 2].mean() > image[..., 0].mean() / image[..., 2].mean()
    flat = apply(image, {"saturation": -1}, OPS)
    assert np.ptp(flat, axis=-1).mean() < np.ptp(image, axis=-1).mean()


def test_regions_follow_mask():
    image = photo()
    mask = np.zeros(image.shape[:2], np.float32)
    mask[:, :48] = 1
    out = apply_regions(image, {"product": {"exposure_ev": 0.5}}, mask, OPS)
    assert np.abs(out[:, :48] - image[:, :48]).mean() > 0.02
    assert np.abs(out[:, 60:] - image[:, 60:]).max() < 1e-3


def test_clamp_and_category_bounds():
    rule = category_rules("product")
    assert rule["bounds"]["temperature"] == [-0.3, 0.3]
    assert rule["fidelity"]["max_mean_delta_e"] < SETTINGS["fidelity"]["max_mean_delta_e"]
    assert clamp_params({"temperature": 2, "made_up": 1, "contrast": float("nan")}, rule["bounds"]) == {"temperature": 0.3}
    for name in load_categories():
        assert category_rules(name)["hint"]


def test_auto_params_fix_blue_cast():
    image = np.clip(photo() * np.array([0.8, 0.95, 1.2], np.float32), 0, 1)
    params, reasons = auto_params(image, SETTINGS["auto"], OPS, category_rules("product")["bounds"])
    assert params["temperature"] > 0 and "temperature" in reasons


def test_fidelity_passes_tone_change_but_catches_drawn_content():
    image = photo()
    rule = category_rules("product")["fidelity"]
    assert check(image, apply(image, {"exposure_ev": 0.1}, OPS), None, rule)["ok"]
    drawn = image.copy()
    drawn[10:40, 20:60] = [0.9, 0.1, 0.1]
    result = check(image, drawn, None, rule)
    assert not result["ok"] and result["problems"]


def test_ai_answer_is_cleaned():
    from models.generative_ai.photo_retouch.interpret import clean_plan, parse_json

    answer = 'sure ```json\n{"global": {"exposure_ev": 3, "bogus": 1}, "product": {"clarity": 0.2}, "reasons": {"global": {"exposure_ev": "어두움"}, "background": {"saturation": 1}},' \
             ' "distractions": [{"label": "컵", "box": [100, 100, 200, 250]}, {"label": "깨짐", "box": [5, 5]}]}\n```'
    plan = clean_plan(parse_json(answer), category_rules("food")["bounds"])
    assert plan["global"] == {"exposure_ev": 1.0} and plan["product"] == {"clarity": 0.2}
    assert [d["label"] for d in plan["distractions"]] == ["컵"] and plan["reasons"] == {"global": {"exposure_ev": "어두움"}, "background": {}}
    assert parse_json("no json") == {}


def test_cleanup_box_and_light_gradient():
    from models.generative_ai.photo_retouch.cleanup import box_pixels
    from models.generative_ai.photo_retouch.studio_light import light_gradient, size_for

    assert box_pixels([0, 0, 500, 500], 200, 100, 0) == (0, 0, 100, 50)
    assert box_pixels([900, 900, 1000, 1000], 200, 100, 1.0) == (160, 70, 200, 100)
    left = light_gradient("left", 8, 4)
    assert left[0, 0, 0] == 1 and left[0, -1, 0] == 0 and light_gradient("none", 8, 4) is None
    assert light_gradient("bottom", 8, 4)[-1, 0, 0] == 1
    assert size_for(2000, 1344, 768) == (768, 512)


def test_explain_mentions_changes():
    from models.generative_ai.photo_retouch.__main__ import explain

    report = {"category_label": "음식", "subject": "도미", "problems": ["어두움"], "chosen": "suggest_x1", "chosen_reason": "자연스러움",
              "params": {"global": {"exposure_ev": 0.3}}, "reasons": {"global": {"exposure_ev": "어두워서 밝혔습니다"}},
              "candidates": [{"name": "suggest_x1", "fidelity": {"ok": True, "delta_e": 3.0, "structure": 0.99, "problems": []}}]}
    text = explain(report)
    assert "밝기 +0.30: 어두워서 밝혔습니다" in text and "통과" in text


def test_saturated_color_is_not_clipping():
    orange = np.zeros((32, 32, 3), np.float32) + np.array([0.9, 0.5, 0.2], np.float32)
    deeper = np.zeros_like(orange) + np.array([0.9, 0.45, 0.0], np.float32)
    assert check(orange, deeper, None, SETTINGS["fidelity"])["new_clipping"] == 0


def test_only_lost_detail_counts_as_clipping():
    rule = SETTINGS["fidelity"]
    sparkle = np.full((10, 10, 3), 0.97, np.float32)
    assert check(sparkle, np.ones_like(sparkle), None, rule)["new_clipping"] == 0
    gray = np.full((10, 10, 3), 0.5, np.float32)
    assert check(gray, np.ones_like(gray), None, rule)["new_clipping"] == 1
    assert check(gray, np.zeros_like(gray), None, rule)["new_clipping"] == 1
