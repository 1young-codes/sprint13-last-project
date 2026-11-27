import json
import unittest
from pathlib import Path
from unittest.mock import patch

from openai import OpenAIError

from models.generative_ai.llm.usage_suggestions import generate_usage_suggestions


class UsageSuggestionsTest(unittest.TestCase):
    def setUp(self):
        module = "models.generative_ai.llm.usage_suggestions"
        openai_patch = patch(f"{module}.OpenAI")
        self.openai = openai_patch.start()
        self.addCleanup(openai_patch.stop)
        self.create = self.openai.return_value.__enter__.return_value.responses.create
        self.create.return_value.status = "completed"

    def test_recommends_for_product_name_only(self):
        expected = ["회", "소금구이", "맑은탕"]
        self.create.return_value.output_text = json.dumps({"suggestions": expected})

        self.assertEqual(generate_usage_suggestions(" 참도미 "), expected)

        self.openai.assert_called_once_with(max_retries=0)
        self.create.assert_called_once()
        request = self.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-5-mini")
        self.assertEqual(json.loads(request["input"]), {"product_name": "참도미"})
        prompt = Path(__file__).resolve().parents[1] / "prompts" / "usage_suggestions.md"
        self.assertEqual(request["instructions"], prompt.read_text(encoding="utf-8"))

    def test_cleans_and_limits_suggestions(self):
        self.create.return_value.output_text = json.dumps(
            {"suggestions": ["  ", " 소금 구이 ", "소금  구이", "회", "맑은탕", "찜"]}
        )

        self.assertEqual(generate_usage_suggestions("참도미"), ["소금 구이", "회", "맑은탕"])

    def test_skips_model_for_empty_name(self):
        for name in ("", " \t\n"):
            with self.subTest(name=name):
                self.assertEqual(generate_usage_suggestions(name), [])
        self.openai.assert_not_called()

    def test_request_failure_returns_empty_without_logging_error_contents(self):
        self.create.side_effect = OpenAIError("secret-in-error-message")

        with self.assertLogs("models.generative_ai.llm.usage_suggestions", "WARNING") as logs:
            self.assertEqual(generate_usage_suggestions("참도미"), [])
        self.assertIn("OpenAIError", logs.output[0])
        self.assertNotIn("secret-in-error-message", logs.output[0])

    def test_rejects_invalid_responses(self):
        for status, output in (
            ("incomplete", '{"suggestions": ["회"]}'),
            ("completed", ""),
            ("completed", "not JSON"),
            ("completed", '["회"]'),
            ("completed", "{}"),
            ("completed", '{"suggestions": "회"}'),
            ("completed", '{"suggestions": ["회", 3]}'),
        ):
            with self.subTest(status=status, output=output), self.assertLogs():
                self.create.return_value.status = status
                self.create.return_value.output_text = output
                self.assertEqual(generate_usage_suggestions("참도미"), [])
