"""원본 사진 → 해석 모델 두 번 → 생성할 장면.

1. 인식: 사진을 보고 객체가 무엇인지 생성 모델이 다시 그릴 수 있을 만큼 자세히 영어로 묘사합니다.
2. 분류와 장면: 그 묘사만 보고 업종을 고르고, 업종별 장면 방향(categories.toml의 example_scene)에 맞는
   활용 예시 후보를 여러 개 냅니다. 첫 번째가 가장 어울리는 것입니다.
   생성 모델도 사진 없이 이 묘사만 보므로, 2번도 사진 없이 묘사만 보게 해서 둘이 같은 정보를 쓰게 합니다.
"""

from PIL import Image

from ..photo_retouch.adjustments import to_array
from ..photo_retouch.interpret import parse_json
from .settings import load_categories, load_settings, read_toml

OPTIONS = 3  # 장면 후보 수
FALLBACK_CATEGORY = "product"


def category_listing(categories: dict) -> str:
    return "\n".join(f"- {key}: {rule['hint']} Scene direction: {rule['example_scene']}"
                     for key, rule in categories.items())


def clean_plan(seen: dict, answer: dict, categories: dict) -> dict:
    """두 답을 합쳐 다듬습니다. 모르는 업종은 상품으로, 빈 후보는 버립니다. 후보가 하나도 없으면 오류."""
    category = answer.get("category") if answer.get("category") in categories else FALLBACK_CATEGORY
    options = [{"name_ko": str(o.get("name_ko") or "").strip(), "prompt": o["prompt"].strip()}
               for o in answer.get("options") or []
               if isinstance(o, dict) and isinstance(o.get("prompt"), str) and o["prompt"].strip()]
    if not options:
        raise ValueError(f"해석 모델이 장면 후보를 내지 않았습니다: {answer}")
    return {"name_ko": str(seen.get("name_ko") or ""), "state": str(seen.get("state") or ""),
            "description": str(seen.get("description") or ""), "category": category,
            "category_label": categories[category]["label"], "options": options}


class Planner:
    def __init__(self):
        from ..photo_retouch.interpret import Interpreter

        self.ai = Interpreter()  # 보정 기능의 OpenAI 호출을 그대로 쓰고, 모델과 이미지 설정만 이 기능 것으로 바꿉니다
        self.ai.settings = self.ai.settings | load_settings()["interpret"]
        self.prompts = read_toml("prompts.toml")

    def plan(self, photo: Image.Image) -> dict:
        seen = parse_json(self.ai.ask("recognize", [to_array(photo)], self.prompts["recognize"]["text"]))
        if not seen.get("description"):
            raise ValueError(f"해석 모델이 사진을 묘사하지 못했습니다: {seen}")
        categories = load_categories()
        text = self.prompts["plan"]["text"].format(name_ko=seen.get("name_ko", ""), state=seen.get("state", ""),
                                                   description=seen["description"], options=OPTIONS,
                                                   categories=category_listing(categories))
        return clean_plan(seen, parse_json(self.ai.ask("plan", [], text)), categories)
