"""Qwen3-VL로 사진을 진단하고, 보정 값을 제안하고, 후보 중 하나를 고릅니다.

VLM은 값과 이유만 냅니다. 픽셀은 adjustments.py가 바꾸므로, VLM이 무엇을 말해도 사진 내용이 새로 그려지지 않습니다.
VLM이 낸 값은 업종 범위로 잘라 쓰고, 모르는 키나 깨진 답은 버립니다.
"""

import json

import numpy as np

from .adjustments import clamp_params, to_image
from .settings import load_categories, load_settings, read_toml

REGIONS = ("global", "product", "background")


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


def clean_plan(raw: dict, bounds: dict) -> dict:
    """VLM 답을 쓸 수 있는 보정 계획으로 다듬습니다."""
    plan = {region: clamp_params(raw.get(region) or {}, bounds) for region in REGIONS}
    reasons = raw.get("reasons") if isinstance(raw.get("reasons"), dict) else {}
    plan["reasons"] = {k: v for k, v in reasons.items() if isinstance(v, str)}
    plan["problems"] = [str(p) for p in raw.get("problems") or [] if isinstance(p, str)]
    boxes = []
    for item in raw.get("distractions") or []:
        box = item.get("box") if isinstance(item, dict) else None
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x0, y0, x1, y1 = (float(np.clip(v, 0, 1000)) for v in box)
            if x1 > x0 and y1 > y0:
                boxes.append({"label": str(item.get("label", "")), "box": [x0, y0, x1, y1]})
    plan["distractions"] = boxes
    return plan


def describe_ranges(bounds: dict) -> str:
    return "\n".join(f"- {key}: {low:+.2f} to {high:+.2f}" for key, (low, high) in bounds.items())


class Vlm:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from .download_models import model_folder

        self.torch = torch
        self.settings = load_settings()["vlm"]
        self.prompts = read_toml("prompts.toml")
        folder = model_folder("vlm")
        self.processor = AutoProcessor.from_pretrained(folder)
        self.model = AutoModelForImageTextToText.from_pretrained(folder, dtype=torch.bfloat16, device_map="cuda").eval()

    def ask(self, images: list[np.ndarray], text: str) -> str:
        long_side = self.settings["image_long_side"]
        pictures = []
        for image in images:
            picture = to_image(image)
            picture.thumbnail((long_side, long_side))
            pictures.append({"type": "image", "image": picture})
        messages = [{"role": "user", "content": [*pictures, {"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.settings["max_new_tokens"], do_sample=False)
        return self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def classify(self, image: np.ndarray) -> tuple[str, str]:
        categories = load_categories()
        listing = "\n".join(f"- {key}: {rule['hint']}" for key, rule in categories.items())
        answer = parse_json(self.ask([image], self.prompts["classify"]["text"].format(categories=listing)))
        category = answer.get("category") if answer.get("category") in categories else self.settings["fallback_category"]
        return category, str(answer.get("subject") or categories[category]["label"])

    def diagnose(self, image: np.ndarray, rule: dict, subject: str) -> tuple[dict, str]:
        text = self.prompts["diagnose"]["text"].format(label=rule["label"], hint=rule["hint"], subject=subject,
                                                       ranges=describe_ranges(rule["bounds"]))
        answer = self.ask([image], text)
        return clean_plan(parse_json(answer), rule["bounds"]), answer

    def judge(self, original: np.ndarray, candidates: list[np.ndarray], rule: dict, subject: str) -> tuple[int, str]:
        """후보 번호(0부터)와 한국어 이유. 답이 깨지면 첫 후보."""
        last = len(candidates) + 1
        text = self.prompts["judge"]["text"].format(last=last, label=rule["label"], subject=subject)
        answer = parse_json(self.ask([original, *candidates], text))
        choice = answer.get("choice")
        if isinstance(choice, int) and 2 <= choice <= last:
            return choice - 2, str(answer.get("reason", ""))
        return 0, ""

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
