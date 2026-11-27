import json
import logging
from pathlib import Path

from openai import APIStatusError, OpenAI, OpenAIError


logger = logging.getLogger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "usage_suggestions.md"
MAX_SUGGESTIONS = 3


def generate_usage_suggestions(product_name: str) -> list[str]:
    """품목명에 어울리는 한국어 활용 예시를 최대 세 개 반환합니다."""
    product_name = product_name.strip()
    if not product_name:
        return []

    try:
        instructions = PROMPT_PATH.read_text(encoding="utf-8")
        with OpenAI(max_retries=0) as client:
            response = client.responses.create(
                model="gpt-5-mini",
                instructions=instructions,
                input=json.dumps({"product_name": product_name}, ensure_ascii=False),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "usage_suggestions",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "suggestions": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["suggestions"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
    except APIStatusError as error:
        logger.warning("활용 예시 추천 요청 실패 (HTTP %s)", error.status_code)
        return []
    except (OpenAIError, OSError, UnicodeError) as error:
        logger.warning("활용 예시 추천 요청 실패 (%s)", type(error).__name__)
        return []

    if response.status != "completed" or not response.output_text:
        logger.warning("활용 예시 추천 응답이 완료되지 않았거나 비어 있습니다.")
        return []

    try:
        result = json.loads(response.output_text)
    except (TypeError, ValueError):
        logger.warning("활용 예시 추천 응답을 JSON으로 읽지 못했습니다.")
        return []

    raw = result.get("suggestions") if isinstance(result, dict) else None
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        logger.warning("활용 예시 추천 응답에 문자열 목록이 없습니다.")
        return []

    cleaned = (" ".join(item.split()) for item in raw)
    return list(dict.fromkeys(item for item in cleaned if item))[:MAX_SUGGESTIONS]
