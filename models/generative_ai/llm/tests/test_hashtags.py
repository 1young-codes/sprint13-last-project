import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from models.generative_ai.llm import hashtags
from models.generative_ai.llm.hashtags import clean_hashtags, generate_hashtags

DOMI = {
    "product_name": "자연산 도미",
    "usage_example": "회, 소금구이",
    "product_caption": "당일 조업한 자연산 도미입니다.",
    "usage_caption": "도톰하게 썬 도미회로 저녁상을 차려 보세요.",
}


def gemini_reply(result: dict) -> dict:
    """Gemini 응답 모양을 흉내 냅니다."""
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(result, ensure_ascii=False)}]}}]}


@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": ""})
class GenerateHashtagsTest(unittest.TestCase):
    @patch.object(hashtags, "_call_gemini")
    def test_returns_hashtags_for_both_posts(self, call):
        call.return_value = gemini_reply(
            {"product_photo": ["#자연산도미", "#도미"], "usage_example": ["#도미회", "#도미소금구이"]}
        )

        result = generate_hashtags(**DOMI)

        self.assertEqual(result, {"product_photo": ["#자연산도미", "#도미"], "usage_example": ["#도미회", "#도미소금구이"]})
        payload, api_key, model, timeout_s = call.call_args.args
        self.assertEqual(api_key, "test-key")
        self.assertEqual(model, hashtags.DEFAULT_MODEL)
        self.assertEqual(timeout_s, hashtags.DEFAULT_TIMEOUT_S)
        self.assertEqual(json.loads(payload["contents"][0]["parts"][0]["text"]), DOMI)
        self.assertIn("hashtags", payload["systemInstruction"]["parts"][0]["text"])

    @patch.object(hashtags, "_call_gemini")
    def test_model_can_be_changed_by_env(self, call):
        call.return_value = gemini_reply({"product_photo": [], "usage_example": []})
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-other"}):
            generate_hashtags(**DOMI)
        self.assertEqual(call.call_args.args[2], "gemini-other")

    @patch.object(hashtags.time, "sleep")
    @patch.object(hashtags, "_call_gemini")
    def test_returns_empty_lists_when_api_fails(self, call, sleep):
        for error in (urllib.error.URLError("down"), TimeoutError(), ValueError("bad json")):
            call.side_effect = error
            self.assertEqual(generate_hashtags(**DOMI), {"product_photo": [], "usage_example": []})

    @patch.object(hashtags.time, "sleep")
    @patch.object(hashtags, "_call_gemini")
    def test_retries_when_gemini_is_busy_or_slow(self, call, sleep):
        busy = urllib.error.HTTPError(hashtags.API_URL, 503, "Service Unavailable", None, None)
        for temporary in (busy, TimeoutError("timed out")):
            call.reset_mock()
            sleep.reset_mock()
            call.side_effect = [temporary, gemini_reply({"product_photo": ["#도미"], "usage_example": ["#도미회"]})]

            self.assertEqual(generate_hashtags(**DOMI), {"product_photo": ["#도미"], "usage_example": ["#도미회"]})
            self.assertEqual(call.call_count, 2)
            sleep.assert_called_once_with(hashtags.RETRY_DELAY_S)

    @patch.object(hashtags.time, "sleep")
    @patch.object(hashtags, "_call_gemini")
    def test_gives_up_after_max_retries(self, call, sleep):
        call.side_effect = urllib.error.HTTPError(hashtags.API_URL, 503, "Service Unavailable", None, None)
        self.assertEqual(generate_hashtags(**DOMI), {"product_photo": [], "usage_example": []})
        self.assertEqual(call.call_count, hashtags.MAX_RETRIES + 1)

    @patch.object(hashtags.time, "sleep")
    @patch.object(hashtags, "_call_gemini")
    def test_does_not_retry_limit_or_key_errors(self, call, sleep):
        for code in (400, 403, 429):
            call.reset_mock()
            call.side_effect = urllib.error.HTTPError(hashtags.API_URL, code, "error", None, None)
            self.assertEqual(generate_hashtags(**DOMI), {"product_photo": [], "usage_example": []})
            self.assertEqual(call.call_count, 1)
        sleep.assert_not_called()

    @patch.object(hashtags, "_call_gemini")
    def test_returns_empty_lists_when_reply_is_strange(self, call):
        for reply in ({}, {"candidates": []}, {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}):
            call.return_value = reply
            self.assertEqual(generate_hashtags(**DOMI), {"product_photo": [], "usage_example": []})

    @patch.object(hashtags, "_call_gemini")
    def test_missing_post_becomes_empty_list(self, call):
        call.return_value = gemini_reply({"product_photo": ["#도미"]})
        self.assertEqual(generate_hashtags(**DOMI), {"product_photo": ["#도미"], "usage_example": []})

    @patch.object(hashtags, "_call_gemini")
    def test_does_not_call_api_without_key_or_product(self, call):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            self.assertEqual(generate_hashtags(**DOMI), {"product_photo": [], "usage_example": []})
        self.assertEqual(generate_hashtags("  ", "회"), {"product_photo": [], "usage_example": []})
        call.assert_not_called()


class CleanHashtagsTest(unittest.TestCase):
    def test_adds_hash_and_removes_spaces_and_symbols(self):
        self.assertEqual(clean_hashtags(["도미 회", "#제철 생선!", "##집밥"]), ["#도미회", "#제철생선", "#집밥"])

    def test_removes_banned_duplicate_and_too_long_tags(self):
        raw = ["#최고의도미", "#원기회복", "#무첨가", "#도미", "#도미", "#" + "가" * 21, 3, None]
        self.assertEqual(clean_hashtags(raw), ["#도미"])

    def test_keeps_at_most_six(self):
        raw = [f"#태그{i}" for i in range(10)]
        self.assertEqual(clean_hashtags(raw), [f"#태그{i}" for i in range(6)])

    def test_non_list_becomes_empty(self):
        self.assertEqual(clean_hashtags("#도미"), [])


if __name__ == "__main__":
    unittest.main()
