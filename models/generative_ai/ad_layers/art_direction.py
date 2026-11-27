"""광고 목적과 상품의 실제 특징에 맞는 연출 가이드를 고릅니다."""

from .creative_brief import ask_stage, required_text


# 사용 조건과 출처는 art_direction_guides.md에 기록합니다.
THEMES = {
    "studio": {
        "when": "The item's shape, material, layers or visible detail is the main evidence.",
        "layout": "Build one axis from the item's shape, light, or edge into the headline. Place the supporting "
                  "name and price together on a quiet alignment. Use real weight and scale contrast within a type "
                  "family before adding another face. Give the strongest visible feature breathing room.",
        "review": "Find the line, light, or material that makes the headline and product one composition. "
                  "A generic product photo with spare text next to it does not establish that relationship.",
    },
    "lifestyle": {
        "when": "A believable gesture, place, or use moment is the evidence for the message.",
        "layout": "Let the gesture, gaze, or direction of light guide the eye to one headline group. "
                  "Use surroundings for context without making them compete. Align the name and price together.",
        "review": "Identify the visible action and its physical support. Check whether that action leads to "
                  "the headline and explains its message. Check that the surroundings do not steal the first glance.",
    },
    "promotion": {
        "when": "An event, seasonal occasion, or specific offer is confirmed in the input.",
        "layout": "Give the confirmed offer one strong typographic contrast through size, weight or color. "
                  "Keep the product and offer in the same reading path. Keep amounts and units together. "
                  "Treat other details as one quieter group and add no invented discount, date or badge.",
        "review": "Check whether the actual offer and product identify each other on the first scan. "
                  "Check that the emphasis belongs to confirmed information and that secondary details stay legible.",
    },
}

ANGLES = {
    "level": "Shoot near the subject's height to show vertical layers, an upright label or the silhouette. "
             "Keep verticals straight and show how the item meets its surface.",
    "three_quarter": "Shoot obliquely from above so the front and top are both visible. Keep the evidence away "
                     "from a rim or edge that would hide it.",
    "overhead": "Shoot straight down when the placement of surfaces or ingredients tells the story. Separate "
                "adjacent shapes with light or spacing and keep the important arrangement in frame.",
    "low": "Shoot slightly below the subject's midpoint for a deliberate sense of height. Use a working distance "
           "that keeps the shape and packaging credible and show a clear contact surface.",
}

SHOT_SCALES = {
    "whole": "Show all essential product edges and visible parts.",
    "close": "Move in to make the visible layer, texture or action prominent while keeping the item recognizable.",
    "detail": "Use a deliberate detail crop only when the advertised feature is recognizable without missing edges or labels.",
}

CHOOSE = """Act as the art director for one Korean ad. The first image is the actual product. Later images
are composition references only. Pick one theme, one angle, and one shot_scale from the offered guides.
Choose by what the camera must reveal, not by category. If the product is shown in use, make the gesture
physically possible. Do not invent product claims, packaging, dates, or offers. For a requested correction,
choose options that preserve unaffected parts of the approved composition.
Return a JSON object with theme, angle, shot_scale, evidence (a specific visible feature in Korean),
and photo_device (one English instruction describing a physical way to reveal the evidence through
framing, light, material or gesture). The device should give the photo a meaningful relationship
to the exact headline. Evidence must be visible in the actual product image. A light, action or prop
that does not yet exist belongs only in photo_device. Include no lettering or graphics in photo_device."""


def plan_art_direction(ai, fields, brief, idea, images, revision=None):
    answer = ask_stage(ai, "ad_art_direction", CHOOSE,
                       {"product_facts": fields, "brief": brief, "idea": idea, "revision": revision,
                        "themes": {key: guide["when"] for key, guide in THEMES.items()},
                        "angles": ANGLES, "shot_scales": SHOT_SCALES}, images)
    plan = {key: required_text(answer, key, "광고 연출")
            for key in ("theme", "angle", "shot_scale", "evidence", "photo_device")}
    for key, options in (("theme", THEMES), ("angle", ANGLES), ("shot_scale", SHOT_SCALES)):
        if plan[key] not in options:
            raise ValueError(f"광고 연출의 {key} 항목은 가이드에 있는 선택지여야 합니다")
    return plan
