import json
import sys

from PIL import Image
import pytest

from models.generative_ai.ad_layers import compare_reference_ads, creative_brief


def test_briefs_flow_from_distinct_product_inputs_to_generation(tmp_path, monkeypatch):
    items = [
        {"file": "a.png", "fields": {"product_name": "무화과 잼", "price_krw": 7900}},
        {"file": "b.png", "fields": {"product_name": "립밤", "price_krw": 25000}},
    ]
    photo_colors = [(180, 0, 0), (0, 170, 0)]
    for item, color in zip(items, photo_colors, strict=True):
        Image.new("RGB", (20, 12), color).save(tmp_path / item["file"])

    stages = ("ad_plan", "ad_direction", "ad_headline", "ad_layout")
    calls = []

    class Interpreter:
        def ask(self, step, images, text, *, model):
            assert model == "gpt-6-luna"
            image_index, stage_index = divmod(len(calls), len(stages))
            assert step == stages[stage_index]
            calls.append(step)
            data = json.loads(text.split("Input JSON follows:\n", 1)[1])
            title = items[image_index]["fields"]["product_name"]
            headline, price = ("첫 문구", "7,900원") if image_index == 0 else ("둘째 문구", "25,000원")
            plan = {"message": f"{title}의 장점", "evidence": "사진에서 확인한 근거"}
            scene = f"studio light {image_index}"

            if step in ("ad_plan", "ad_direction"):
                assert len(images) == 1 and images[0][0, 0].argmax() == image_index
            else:
                assert not images
            if step == "ad_plan":
                assert data["product_name"] == title and "plan" not in data
                return json.dumps(plan)
            if step == "ad_direction":
                assert data["plan"] == plan and "scene" not in data
                return json.dumps({"scene": scene})
            if step == "ad_headline":
                assert data["plan"] == plan and data["scene"] == scene and "copy" not in data
                return json.dumps({"headline": headline})
            assert data["scene"] == scene and data["message"] == plan["message"]
            assert data["copy"] == {"headline": headline, "product_name": title, "price": price}
            return json.dumps({
                "boxes": {
                    "product": [0, 20, 45, 60], "headline": [50, 15, 45, 25],
                    "product_name": [50, 60, 30, 15], "price": [80, 60, 15, 15],
                },
                "reading_order": ["product", "headline", "product_name", "price"],
                "typography": "clear",
            })

    plan = creative_brief.prepare_ads(Interpreter(), items, tmp_path)
    assert calls == list(stages) * len(items)
    expected = [ad["prompt"] for ad in plan["ads"]]
    assert "첫 문구" in expected[0] and "7,900원" in expected[0]
    assert "둘째 문구" in expected[1] and "25,000원" in expected[1]
    assert all('"headline": [50, 15, 45, 25]' in prompt for prompt in expected)
    instructions = tmp_path / "plan.json"
    instructions.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "ads"

    def generate(args, jobs):
        assert [prompt for prompt, _ in jobs] == expected
        assert [image.getpixel((0, 0)) for _, image in jobs] == photo_colors
        return (image for _, image in jobs)

    monkeypatch.setattr(compare_reference_ads, "generate", generate)
    monkeypatch.setattr(sys, "argv", ["compare", "flux", str(instructions), str(tmp_path), str(output), str(tmp_path)])
    compare_reference_ads.main()
    settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
    assert settings["ads"] == plan["ads"]
    assert all((output / ad["file"]).is_file() for ad in plan["ads"])


def test_brainstorm_stops_when_earlier_stage_is_incomplete():
    class Interpreter:
        def ask(self, step, images, text, *, model):
            assert step == "ad_plan"
            return '{"message": "메시지"}'

    with pytest.raises(ValueError, match="기획 단계에서 evidence"):
        creative_brief.brainstorm(Interpreter(), Image.new("RGB", (20, 12)), {"product_name": "필통"}, "", "5,000원")


def test_layout_rejects_text_box_extending_past_canvas():
    answer = {"boxes": {role: [0, 0, 10, 10] for role in creative_brief.LAYOUT_ROLES}}
    answer["boxes"]["price"] = [85, 80, 20, 10]
    with pytest.raises(ValueError, match="price .*캔버스 범위"):
        creative_brief.layout_boxes(answer)
