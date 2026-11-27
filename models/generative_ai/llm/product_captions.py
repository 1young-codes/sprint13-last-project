import json
import os
from pathlib import Path

from openai import OpenAI


CAPTION_PROPERTIES = {
    "product_caption": {"type": "string"},
    "usage_caption": {"type": "string"},
}

DEFAULT_MODEL = "gpt-5-mini"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "product_captions.md"


def generate_product_captions(
    product_name: str,
    unit: str,
    price_krw: int,
    usage_example: str,
    notes: str = "",
) -> dict[str, str]:
    """판매자가 입력한 상품 정보로 한국어 상품 소개와 활용 원고를 생성합니다."""
    for field, value in (
        ("product_name", product_name),
        ("unit", unit),
        ("usage_example", usage_example),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field}에 비어 있지 않은 문자열이 필요합니다.")
    if isinstance(price_krw, bool) or not isinstance(price_krw, int) or price_krw <= 0:
        raise ValueError("price_krw는 양의 정수여야 합니다.")
    if not isinstance(notes, str):
        raise ValueError("notes는 문자열이어야 합니다.")

    unit = unit.strip()
    display_unit = "1kg" if unit == "kg" else unit
    price_label = f"{display_unit}당 {price_krw:,}원"

    model = os.environ.get("OPENAI_CAPTION_MODEL", "").strip() or DEFAULT_MODEL
    effort = os.environ.get("OPENAI_CAPTION_REASONING_EFFORT", "").strip()
    reasoning_options = {"reasoning": {"effort": effort}} if effort else {}
    instructions = PROMPT_PATH.read_text(encoding="utf-8")

    # OpenAI SDK reads OPENAI_API_KEY from the environment.
    with OpenAI(max_retries=0) as client:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(
                {
                    "product_name": product_name.strip(),
                    "unit": unit,
                    "price_krw": price_krw,
                    "usage_example": usage_example.strip(),
                    "notes": notes.strip(),
                },
                ensure_ascii=False,
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "product_captions",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": CAPTION_PROPERTIES,
                        "required": list(CAPTION_PROPERTIES),
                        "additionalProperties": False,
                    },
                }
            },
            **reasoning_options,
        )

    if response.status != "completed" or not response.output_text:
        raise ValueError("상품 문구 생성 응답을 완료하지 못했습니다.")

    captions = json.loads(response.output_text)
    if not isinstance(captions, dict):
        raise ValueError("상품 문구 생성 응답은 JSON 객체여야 합니다.")
    for field in CAPTION_PROPERTIES:
        value = captions.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field}에 비어 있지 않은 문구가 필요합니다.")
    captions["product_caption"] = f"{captions['product_caption'].rstrip()}\n{price_label}"
    return captions
