"""그래픽 카드와 OpenAI 없이 활용 예시의 계획, 조건, 평가 페이지를 검증합니다."""

import csv
import json

import pytest
from PIL import Image

from models.generative_ai.usage_examples import plan as plan_module
from models.generative_ai.usage_examples.__main__ import (assign_orientation, calibration_plans, command_plan,
                                                           read_photos)
from models.generative_ai.usage_examples.plan import clean_plan
from models.generative_ai.usage_examples.review_page import CSV_FIELDS, build, summarize
from models.generative_ai.usage_examples.settings import condition_prompt, conditions, load_categories, load_settings


def test_every_category_has_scene_direction():
    for rule in load_categories().values():
        assert rule["example_scene"]


def test_conditions_and_trigger_word():
    names = [c["name"] for c in conditions()]
    assert names == ["original", "lora1", "lora2"]
    lora2 = conditions(["lora2"])[0]
    assert condition_prompt(lora2, "a bowl of bibimbap") == "cndamiph, a bowl of bibimbap"
    assert condition_prompt(conditions(["original"])[0], "a bowl") == "a bowl"
    with pytest.raises(ValueError):
        conditions(["lora9"])


def test_sizes_are_sdxl_3_2_and_multiples_of_16():
    sizes = load_settings()["generation"]["sizes"]
    assert sizes == {"landscape": [1216, 832], "portrait": [832, 1216]}
    assert all(n % 16 == 0 for size in sizes.values() for n in size)


def test_clean_plan_drops_empty_options_and_unknown_category():
    categories = load_categories()
    plan = clean_plan({"name_ko": "고등어", "state": "raw ingredient", "description": "A whole mackerel."},
                      {"category": "spaceship", "options": [{"name_ko": "", "prompt": " "},
                                                            {"name_ko": "고등어조림", "prompt": "Braised mackerel."}]},
                      categories)
    assert plan["category"] == "product"
    assert [o["name_ko"] for o in plan["options"]] == ["고등어조림"]
    with pytest.raises(ValueError):
        clean_plan({}, {"category": "food", "options": []}, categories)


def test_orientation_is_half_and_half_inside_each_group():
    groups = ["raw"] * 8 + ["korean"] * 10 + ["western"] * 5 + ["dessert"] * 4 + ["drink"] * 3  # 실험의 음식 30장
    plans = [{"group": g, "category": "food"} for g in groups]
    assign_orientation(plans)
    sides = [p["orientation"] for p in plans]
    assert sides.count("landscape") == sides.count("portrait") == 15
    for group in set(groups):
        mine = [p["orientation"] for p in plans if p["group"] == group]
        assert abs(mine.count("landscape") - mine.count("portrait")) <= 1


def test_calibration_picks_across_groups():
    plans = [{"id": f"{g}{i}", "group": g, "category": "food"} for g in "abc" for i in range(3)]
    assert [p["id"] for p in calibration_plans(plans, 4)] == ["a0", "b0", "c0", "a1"]


def test_read_photos_uses_sources_list(tmp_path):
    (tmp_path / "sources.toml").write_text('[[photos]]\nid = "x"\nfile = "x.jpg"\ngroup = "food_raw"\n',
                                           encoding="utf-8")
    assert read_photos(tmp_path)[1] == [{"id": "x", "file": "x.jpg", "group": "food_raw"}]


def test_plan_usage_adds_up_across_runs(tmp_path, monkeypatch):
    """다시 실행해도 이전 실행의 사용량이 지워지지 않고 더해져야 합니다."""
    class FakePlanner:
        def __init__(self):
            self.ai = type("AI", (), {"usage": []})()

        def plan(self, photo):
            self.ai.usage += [{"step": "recognize", "input_tokens": 100, "output_tokens": 10}] * 2
            return {"category": "food", "category_label": "음식", "name_ko": "", "options": [{"name_ko": "", "prompt": "p"}]}

    monkeypatch.setattr(plan_module, "Planner", FakePlanner)
    photos, run = tmp_path / "photos", tmp_path / "run"
    photos.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (8, 8)).save(photos / f"{name}.png")
    usage = lambda: json.loads((run / "plans.json").read_text(encoding="utf-8"))["usage"]
    command_plan(photos, run)
    assert usage() == {"calls": 4, "input_tokens": 400, "output_tokens": 40}
    Image.new("RGB", (8, 8)).save(photos / "c.png")
    command_plan(photos, run)  # 새 사진 하나만 해석
    assert usage() == {"calls": 6, "input_tokens": 600, "output_tokens": 60}
    command_plan(photos, run)  # 할 일이 없어도 그대로
    assert usage()["calls"] == 6


def test_review_page_and_summary(tmp_path):
    photos, images, out = tmp_path / "photos", tmp_path / "images", tmp_path / "review"
    photos.mkdir()
    Image.new("RGB", (64, 48)).save(photos / "fish.jpg")
    for condition in conditions():
        (images / condition["name"]).mkdir(parents=True)
        Image.new("RGB", (64, 48)).save(images / condition["name"] / "fish.png")
    plan = {"id": "fish", "file": "fish.jpg", "name_ko": "고등어", "category_label": "음식", "orientation": "landscape",
            "option": {"name_ko": "고등어조림", "prompt": "Braised <mackerel>."}}
    page = build([plan], images, photos, out).read_text(encoding="utf-8")
    assert "Realistic Snapshot" in page and "MOSSRealisticSnapshotV1" not in page and "cndamiph_Candid_Amateur_iPhone_MOSS_v1" in page
    assert "Braised &lt;mackerel&gt;." in page  # 프롬프트는 글자로만 보임
    assert len(list((out / "images").iterdir())) == 3

    for person, order in (("가", ["1", "2", "3"]), ("나", ["2", "1", "3"])):
        with open(tmp_path / f"{person}.csv", "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_FIELDS)
            for condition, rank in zip(conditions(), order):
                writer.writerow([person, 1, "fish", "고등어", condition["label"], condition.get("file", "로라 없음"), rank])
    result = summarize([tmp_path / "가.csv", tmp_path / "나.csv"])
    assert list(result["평균 순위"].values()) == [1.5, 1.5, 3.0]
    assert result["1위 횟수"]["오리지널 (로라 없음)"] == 1
