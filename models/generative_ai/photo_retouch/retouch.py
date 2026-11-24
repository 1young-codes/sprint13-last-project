"""자동 보정 전체 순서를 지휘합니다. 업스케일을 마친 사진에 씁니다(카메라 각도 → 업스케일 → 자동 보정).

1. 상품 영역 분리 (BiRefNet)
2. 사진을 재서 바로잡을 거리(밝기, 색 치우침, 대비, 날아감, 뭉개짐)와 그 값을 계산
3. 해석 모델(OpenAI)이 업종을 판단하고, 잰 문제마다 바로잡을지 분위기라 둘지 판단해 한국어 이유를 냄
4. 후보 만들기: 바로잡기 + 업종별 약한 색감을 세기별로, 그리고 판단 없이 잰 것을 전부 바로잡은 것
5. 충실도 검사로 실물을 바꾼 후보를 탈락시키고, 남은 후보 중 해석 모델이 광고에 가장 맞는 것을 고름 (모두 탈락하면 원본)
6. (선택, 기본 끔) 배경 잡동사니 지우기: 상품 영역 밖만 지움. 해석 모델의 좌표가 부정확할 수 있음
7. (선택) 스튜디오 조명: 새 조명의 밝기만 상품에 입히고, 다시 검사해서 통과할 때만 별도 결과로 냄

모델은 단계마다 불러오고 바로 내려서, 그래픽 카드 L4(24GB) 한 장에서 돕니다.
"""

from .adjustments import apply_regions, clamp_params, scale_params, to_array, to_image
from .auto_params import combine, measure
from .fidelity import check
from .settings import category_rules, load_settings


def make_plan(found: dict, names, rule: dict, calm: dict | None, calm_reason: str, look: bool) -> dict:
    """고른 바로잡기 + (선택) 업종 색감 + (선택) 배경 차분하게 → {"plan": 영역별 값, "reasons": 영역별 이유}."""
    params, reasons = combine(found, names)
    if look:
        for key, value in rule.get("look", {}).items():
            params[key] = params.get(key, 0.0) + value
            reasons[key] = rule["look_reason"] if key not in reasons else f"{reasons[key]} {rule['look_reason']}"
    plan, why = {"global": clamp_params(params, rule["bounds"])}, {"global": reasons}
    if calm:
        plan["background"] = dict(calm)
        why["background"] = {key: calm_reason for key in calm}
    return {"plan": plan, "reasons": why}


def build_candidates(image, found, review, rule, mask, settings) -> list[dict]:
    if review is None:  # 해석 없이: 잰 것을 전부 바로잡음
        names, calm, calm_reason = list(found), None, ""
    else:
        names = list(review["fix"])
        for name in names:
            found[name] = found[name] | {"reason": review["fix"][name] or found[name]["reason"]}
        calm_reason = review["calm_background"]
        calm = settings["auto"]["calm_background"] if calm_reason and mask is not None else None
    base = make_plan(found, names, rule, calm, calm_reason, look=True)
    candidates = [{"name": f"corrected_x{strength:g}", "reasons": base["reasons"],
                   "plan": {region: scale_params(values, strength) for region, values in base["plan"].items()}}
                  for strength in settings["candidates"]["strengths"]]
    if review is not None and settings["candidates"]["include_measured"]:
        candidates.append({"name": "measured"} | make_plan(found, list(found), rule, None, "", look=False))
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

    ai, review, subject = None, None, ""
    if use_ai:
        from .interpret import Interpreter

        ai = Interpreter()
        if category is None:
            category, subject = ai.classify(image)
    category = category or settings["interpret"]["fallback_category"]
    rule = category_rules(category)
    mask = subject_mask if rule["use_product_mask"] else None
    subject = subject or rule["label"]
    found = measure(image, settings["auto"], settings["ops"], rule["bounds"], mask)
    report |= {"category": category, "category_label": rule["label"], "subject": subject,
               "measured": {name: {"params": f["params"], "reason": f["reason"]} for name, f in found.items()}}
    if ai:
        review, raw_answer = ai.review(image, rule, subject, found)
        report |= {"problems": review["problems"], "kept": {name: why for name, why in review["keep"].items()},
                   "ai_answer": raw_answer}

    candidates = build_candidates(image, found, review, rule, mask, settings)
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

    if cleanup and review and review["distractions"]:
        from .cleanup import remove_distractions

        cleaned, removed = remove_distractions(result, review["distractions"], subject_mask, settings["cleanup"])
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
