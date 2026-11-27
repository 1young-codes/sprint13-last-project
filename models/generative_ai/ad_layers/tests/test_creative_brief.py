import json
import sys

from PIL import Image

from models.generative_ai.ad_layers import compare_reference_ads, creative_brief


def test_briefs_flow_from_distinct_product_inputs_to_generation(tmp_path, monkeypatch):
    items = [
        {"file": "a.png", "fields": {"product_name": "무화과 잼", "price_krw": 7900}},
        {"file": "b.png", "fields": {"product_name": "립밤", "price_krw": 25000}},
    ]
    photo_colors = [(180, 0, 0), (0, 170, 0)]
    for item, color in zip(items, photo_colors, strict=True):
        Image.new("RGB", (20, 12), color).save(tmp_path / item["file"])

    class Interpreter:
        def ask(self, step, images, text, *, model):
            assert step == "ad_design" and model == "gpt-6-luna"
            data = json.loads(text.split("Product information JSON follows:\n", 1)[1])
            first = data["product_name"] == items[0]["fields"]["product_name"]
            assert images[0][0, 0].argmax() == (0 if first else 1)
            assert data["price"] == ("7,900원" if first else "25,000원")
            return json.dumps({
                "message": "test message", "headline": "첫 문구" if first else "둘째 문구",
                "scene": "side light" if first else "window light",
                "composition": "left" if first else "right", "typography": "clear",
            })

    plan = creative_brief.prepare_ads(Interpreter(), items, tmp_path)
    expected = [ad["prompt"] for ad in plan["ads"]]
    assert "첫 문구" in expected[0] and "7,900원" in expected[0]
    assert "둘째 문구" in expected[1] and "25,000원" in expected[1]
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
