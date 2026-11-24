"""OpenAI 모델로 사진을 해석합니다: 업종 판단, 잰 문제를 바로잡을지 판단, 후보 고르기, 지울 물건 위치.

해석만 합니다. 보정 값은 auto_params.py가 사진을 재서 정하고, 픽셀은 adjustments.py가 바꿉니다.
모르는 이름이나 깨진 답은 버립니다.
"""

import base64
import io
import json

import numpy as np

from .adjustments import to_image
from .settings import env_value, load_categories, load_settings, read_toml


def parse_json(text: str) -> dict:
    """답에서 첫 { 부터 마지막 } 까지를 JSON으로 읽습니다. 못 읽으면 빈 dict."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def clean_review(raw: dict, offered) -> dict:
    """모델 답을 다듬습니다. 측정에서 나온 이름만 받고, 바로잡으라는 답이 없으면 둡니다."""
    def named(key):
        value = raw.get(key) if isinstance(raw.get(key), dict) else {}
        return {name: reason for name, reason in value.items() if name in offered and isinstance(reason, str)}

    fix = named("fix")
    calm = raw.get("calm_background")
    boxes = []
    for item in raw.get("distractions") or []:
        box = item.get("box") if isinstance(item, dict) else None
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x0, y0, x1, y1 = (float(np.clip(v, 0, 1000)) for v in box)
            if x1 > x0 and y1 > y0:
                boxes.append({"label": str(item.get("label", "")), "box": [x0, y0, x1, y1]})
    return {"fix": fix, "keep": {k: v for k, v in named("keep").items() if k not in fix},
            "calm_background": calm.strip() if isinstance(calm, str) else "",
            "problems": [p for p in raw.get("problems") or [] if isinstance(p, str)], "distractions": boxes}


class Interpreter:
    def __init__(self):
        from openai import OpenAI

        self.settings = load_settings()["interpret"]
        self.prompts = read_toml("prompts.toml")
        key = env_value(self.settings["api_key_env"], prefixed=False)
        if not key:
            raise RuntimeError(f"{self.settings['api_key_env']}가 없습니다. 프로젝트 루트 .env에 넣거나 --no-ai로 실행하세요")
        self.client = OpenAI(api_key=key, timeout=self.settings["timeout_seconds"])

    def ask(self, images: list[np.ndarray], text: str) -> str:
        long_side = self.settings["image_long_side"]
        content = []
        for image in images:
            picture = to_image(image)
            picture.thumbnail((long_side, long_side))
            buffer = io.BytesIO()
            picture.save(buffer, "JPEG", quality=self.settings["jpeg_quality"])
            url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": url, "detail": self.settings["detail"]}})
        content.append({"type": "text", "text": text})
        response = self.client.chat.completions.create(model=self.settings["model"], response_format={"type": "json_object"},
                                                       messages=[{"role": "user", "content": content}])
        return response.choices[0].message.content or ""

    def classify(self, image: np.ndarray) -> tuple[str, str]:
        categories = load_categories()
        listing = "\n".join(f"- {key}: {rule['hint']}" for key, rule in categories.items())
        answer = parse_json(self.ask([image], self.prompts["classify"]["text"].format(categories=listing)))
        category = answer.get("category") if answer.get("category") in categories else self.settings["fallback_category"]
        return category, str(answer.get("subject") or categories[category]["label"])

    def review(self, image: np.ndarray, rule: dict, subject: str, found: dict) -> tuple[dict, str]:
        """측정된 바로잡을 거리마다 진짜 문제인지, 분위기라 둘지 판단합니다."""
        findings = "\n".join(f"- {name}: {item['finding']}" for name, item in found.items()) or "- (none)"
        text = self.prompts["review"]["text"].format(label=rule["label"], hint=rule["hint"], subject=subject, findings=findings)
        answer = self.ask([image], text)
        return clean_review(parse_json(answer), found), answer

    def judge(self, original: np.ndarray, candidates: list[np.ndarray], rule: dict, subject: str) -> tuple[int, str]:
        """후보 번호(0부터)와 한국어 이유. 답이 깨지면 첫 후보."""
        last = len(candidates) + 1
        text = self.prompts["judge"]["text"].format(last=last, label=rule["label"], subject=subject)
        answer = parse_json(self.ask([original, *candidates], text))
        choice = answer.get("choice")
        if isinstance(choice, int) and 2 <= choice <= last:
            return choice - 2, str(answer.get("reason", ""))
        return 0, ""
