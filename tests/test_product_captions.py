import json
import unittest
from unittest.mock import patch

from models.product_captions import generate_product_captions


class ProductCaptionsTest(unittest.TestCase):
    @patch("models.product_captions.OpenAI")
    def test_returns_both_captions_from_one_request(self, openai):
        inputs = {
            "product_name": "도미",
            "unit": "kg",
            "price_krw": 25000,
            "usage_example": "도미 솥밥",
            "notes": "제철 도미, 당일 조업, 원기 회복, 건강한 한 끼",
        }
        model_captions = {
            "product_caption": "당일 조업한 제철 도미입니다.",
            "usage_caption": "기운이 빠지는 날, 따뜻한 도미 솥밥에 쪽파를 올려 든든한 저녁을 즐겨 보세요.",
        }
        create = openai.return_value.__enter__.return_value.responses.create
        create.return_value.status = "completed"
        create.return_value.output_text = json.dumps(model_captions, ensure_ascii=False)

        captions = generate_product_captions(**inputs)

        self.assertEqual(
            captions,
            {
                "product_caption": model_captions["product_caption"] + "\n1kg당 25,000원",
                "usage_caption": model_captions["usage_caption"],
            },
        )
        for field in ("product_caption", "usage_caption"):
            self.assertIsInstance(captions[field], str)
            self.assertTrue(captions[field].strip())
        openai.assert_called_once_with(max_retries=0)
        create.assert_called_once()
        request = create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-5-mini")
        self.assertEqual(json.loads(request["input"]), inputs)
        self.assertCountEqual(
            request["text"]["format"]["schema"]["required"],
            ["product_caption", "usage_caption"],
        )

    @patch("models.product_captions.OpenAI")
    def test_preserves_free_form_unit_and_allows_omitting_notes(self, openai):
        create = openai.return_value.__enter__.return_value.responses.create
        create.return_value.status = "completed"
        create.return_value.output_text = json.dumps(
            {"product_caption": "버터 스콘입니다.", "usage_caption": "주말 브런치에 곁들여 보세요."}
        )

        captions = generate_product_captions("스콘", "6개입 한 상자", 18000, "주말 브런치")

        self.assertEqual(captions["product_caption"], "버터 스콘입니다.\n6개입 한 상자당 18,000원")
        self.assertEqual(json.loads(create.call_args.kwargs["input"])["notes"], "")

    @patch("models.product_captions.OpenAI")
    def test_rejects_invalid_input_before_opening_client(self, openai):
        inputs = dict(product_name="도미", unit="kg", price_krw=25000, usage_example="도미 솥밥")
        for field, value in (
            ("product_name", " "),
            ("unit", ""),
            ("usage_example", "\t"),
            ("price_krw", 0),
            ("price_krw", -1),
            ("price_krw", True),
            ("price_krw", 25000.0),
            ("price_krw", "25000"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    generate_product_captions(**(inputs | {field: value}))
        openai.assert_not_called()

    @patch("models.product_captions.OpenAI")
    def test_rejects_empty_missing_or_non_string_caption(self, openai):
        create = openai.return_value.__enter__.return_value.responses.create
        create.return_value.status = "completed"
        for payload, field in (
            ({"product_caption": " ", "usage_caption": "활용 예시"}, "product_caption"),
            ({"product_caption": "상품 소개"}, "usage_caption"),
            ({"product_caption": "상품 소개", "usage_caption": 3}, "usage_caption"),
        ):
            with self.subTest(payload=payload):
                create.return_value.output_text = json.dumps(payload)
                with self.assertRaisesRegex(ValueError, field):
                    generate_product_captions("도미", "kg", 25000, "도미 솥밥")
