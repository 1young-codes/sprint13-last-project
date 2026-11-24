"""자동 보정 전체 순서를 지휘합니다.

1. 상품 영역 분리 (BiRefNet)
2. 해석 모델(OpenAI)이 업종을 판단하고, 사진을 진단해 보정 값과 한국어 이유를 냄
3. 후보 만들기: 해석 모델 제안을 세기별로, 그리고 해석 없이 잰 기본 자동 보정
4. 충실도 검사로 실물을 바꾼 후보를 탈락시키고, 남은 후보 중 해석 모델이 광고에 가장 맞는 것을 고름 (모두 탈락하면 원본)
5. (선택, 기본 끔) 배경 잡동사니 지우기: 상품 영역 밖만 지움. 해석 모델의 좌표가 부정확할 수 있음
6. (선택) 스튜디오 조명: 생성형 재조명 뒤 원본 세부 질감을 다시 입히고, 다시 검사해서 통과할 때만 별도 결과로 냄

모델은 단계마다 불러오고 바로 내려서, 그래픽 카드 L4(24GB) 한 장에서 돕니다.
"""

from .adjustments import apply_regions, scale_params, to_array, to_image
from .auto_params import auto_params
from .fidelity import check
from .settings import category_rules, load_settings


def scale_plan(plan: dict, strength: float) -> dict:
    return {region: scale_params(plan.get(region, {}), strength) for region in ("global", "product", "background")}


def build_candidates(image, plan, rule, mask, settings) -> list[dict]:
    candidates = []
    if plan:
        for strength in settings["candidates"]["strengths"]:
            candidates.append({"name": f"suggest_x{strength:g}", "plan": scale_plan(plan, strength), "reasons": plan["reasons"]})
    if settings["candidates"]["include_auto"]:
        params, reasons = auto_params(image, settings["auto"], settings["ops"], rule["bounds"], mask)
        candidates.append({"name": "auto", "plan": {"global": params}, "reasons": {"global": reasons}})
    for candidate in candidates:
        candidate["image"] = apply_regions(image, candidate["plan"], mask, settings["ops"])
        candidate["fidelity"] = check(image, candidate["image"], mask, rule["fidelity"])
    return candidates


def retouch(photo, category: str | None = None, use_ai: bool = True, cleanup: bool = False,
            studio_light: bool = False) -> dict:
    """photo: PIL 이미지. 결과: {"image", "studio_image"(선택), "report"}."""
    settings = load_settings()
    image = to_array(photo)
    report = {"steps": []}

    from .product_mask import product_mask

    subject_mask = product_mask(image)

    ai = None
    plan, subject = None, ""
    if use_ai:
        from .interpret import Interpreter

        ai = Interpreter()
        if category is None:
            category, subject = ai.classify(image)
    category = category or settings["interpret"]["fallback_category"]
    rule = category_rules(category)
    mask = subject_mask if rule["use_product_mask"] else None
    subject = subject or rule["label"]
    report |= {"category": category, "category_label": rule["label"], "subject": subject}
    if ai:
        plan, raw_answer = ai.diagnose(image, rule, subject)
        report |= {"problems": plan["problems"], "ai_answer": raw_answer}

    candidates = build_candidates(image, plan, rule, mask, settings)
    passing = [c for c in candidates if c["fidelity"]["ok"]]
    report["candidates"] = [{"name": c["name"], "plan": c["plan"], "fidelity": c["fidelity"]} for c in candidates]
    if not passing:
        chosen, why = None, "모든 보정 후보가 실물을 너무 바꿔서 원본을 그대로 씁니다"
    elif ai and len(passing) > 1:
        index, why = ai.judge(image, [c["image"] for c in passing], rule, subject)
        chosen = passing[index]
    else:
        chosen, why = passing[0], ""
    result = chosen["image"] if chosen else image
    report |= {"chosen": chosen["name"] if chosen else "original", "chosen_reason": why,
               "params": chosen["plan"] if chosen else {}, "reasons": chosen["reasons"] if chosen else {}}

    if cleanup and plan and plan["distractions"]:
        from .cleanup import remove_distractions

        cleaned, removed = remove_distractions(result, plan["distractions"], subject_mask, settings["cleanup"])
        verdict = check(result, cleaned, subject_mask, rule["fidelity"])  # 상품 영역은 그대로여야 함
        if removed and verdict["ok"]:
            result = cleaned
            report["removed"] = removed
        report["steps"].append({"cleanup": removed, "fidelity": verdict})

    output = {"image": to_image(result), "report": report}
    if studio_light and rule["use_product_mask"] and rule.get("studio_prompt"):
        from .studio_light import relight

        light = settings["studio_light"]
        lit = relight(result, subject_mask, rule["studio_prompt"], light)
        verdict = check(result, lit, subject_mask, rule["fidelity"] | light["fidelity"])
        report["studio_light"] = verdict
        output["studio_image" if verdict["ok"] else "studio_rejected"] = to_image(lit)
    elif studio_light:
        report["studio_light"] = {"ok": False, "problems": [f"{rule['label']} 사진에는 스튜디오 조명을 쓰지 않습니다"]}
    return output
