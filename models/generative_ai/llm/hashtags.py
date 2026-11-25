"""게시글 해시태그 만들기.

품목, 활용 예시, 문구 두 편을 넣으면 게시글마다 해시태그 목록을 돌려줍니다.
무료 사용량이 있는 Google Gemini API를 부르고, 어떤 이유로든 실패하면 빈 목록을 돌려줍니다.
서버는 문구를 만든 뒤 이 함수를 한 번 부르고, 결과를 API 응답의 hashtags 칸에 넣습니다 (docs/api-spec.md).

API 키는 환경 변수 GEMINI_API_KEY에서 읽습니다. 키는 .env에만 두고 저장소에 올리지 않습니다.
새 라이브러리 없이 파이썬 기본 기능(urllib)만 씁니다.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "hashtags.md"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# 결과가 매번 같은 모델에서 나오도록 이름을 고정합니다. 'latest' 같은 별칭은 모델이 몰래 바뀔 수 있어 쓰지 않습니다.
# gemini-2.5-flash는 새 사용자에게 막혀 있어서(2026-11 확인) 가벼운 3.5 Flash Lite를 씁니다.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT_S = 5.0

POSTS = ("product_photo", "usage_example")
MAX_TAGS = 6
MAX_TAG_LENGTH = 20  # '#' 제외 글자 수

# 모델이 규칙을 어겨도 화면에 나가지 않게 한 번 더 거르는 표현 (과장, 효능, 확인되지 않은 사실)
BANNED_WORDS = ("최고", "최상", "1등", "원기", "회복", "보약", "효능", "치료", "무첨가", "다이어트")


def _empty() -> dict[str, list[str]]:
    return {post: [] for post in POSTS}


def _clean_tag(raw: object) -> str | None:
    """'#' 붙이기, 띄어쓰기와 기호 지우기. 쓸 수 없는 태그면 None."""
    if not isinstance(raw, str):
        return None
    word = re.sub(r"[^0-9A-Za-z가-힣_]", "", raw)
    if not word or len(word) > MAX_TAG_LENGTH:
        return None
    if any(banned in word for banned in BANNED_WORDS):
        return None
    return f"#{word}"


def clean_hashtags(raw_tags: object) -> list[str]:
    """모델이 돌려준 목록을 정리합니다: 형식 맞추기, 금지 표현 빼기, 중복 빼기, 개수 자르기."""
    if not isinstance(raw_tags, list):
        return []
    tags: list[str] = []
    for raw in raw_tags:
        tag = _clean_tag(raw)
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:MAX_TAGS]


def _call_gemini(payload: dict, api_key: str, model: str, timeout_s: float) -> dict:
    request = urllib.request.Request(
        API_URL.format(model=model),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def generate_hashtags(
    product_name: str,
    usage_example: str,
    product_caption: str = "",
    usage_caption: str = "",
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, list[str]]:
    """게시글 두 개의 해시태그 목록을 돌려줍니다.

    반환 예: {"product_photo": ["#자연산도미", "#도미"], "usage_example": ["#도미회"]}
    키가 없거나, 시간이 넘거나, 응답이 이상하면 두 목록 모두 빈 목록입니다. 예외를 내지 않습니다.
    """
    if not str(product_name or "").strip():
        return _empty()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("GEMINI_API_KEY가 없어 해시태그를 만들지 않았습니다.")
        return _empty()
    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL

    product_data = {
        "product_name": str(product_name).strip(),
        "usage_example": str(usage_example or "").strip(),
        "product_caption": str(product_caption or "").strip(),
        "usage_caption": str(usage_caption or "").strip(),
    }
    payload = {
        "systemInstruction": {"parts": [{"text": PROMPT_PATH.read_text(encoding="utf-8")}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(product_data, ensure_ascii=False)}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {post: {"type": "ARRAY", "items": {"type": "STRING"}} for post in POSTS},
                "required": list(POSTS),
            },
        },
    }

    try:
        response = _call_gemini(payload, api_key, model, timeout_s)
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, TypeError, ValueError) as error:
        logger.warning("해시태그를 만들지 못했습니다: %s", error)
        return _empty()

    if not isinstance(result, dict):
        return _empty()
    return {post: clean_hashtags(result.get(post)) for post in POSTS}
