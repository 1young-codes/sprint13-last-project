"""포스터를 보고 찾을 수 있는 작업 가이드입니다."""

START_GUIDES = {
    "product_shoot": "Show the real product and one selling detail through light, scale, or a believable use scene.",
    "reference_poster": "Borrow the reference's composition or hierarchy. Use our product, visual identity and copy.",
    "new_concept": "Build one visual idea from the approved product benefit and requested placement.",
}

ACTION_GUIDES = {
    "edit_text": {
        "findings": ("typography", "layout"),
        "guide": "Change only the typography that needs work. Keep the exact wording and photograph. "
                 "Use shared alignment and nearby spacing for related copy. Keep the heading in charge, "
                 "while giving the product name and price enough size and contrast to read together. "
                 "Move glyphs out of the product and check legibility in the actual image.",
        "arguments": {"changes": "Object keyed by headline, product_name or price. Put edits directly inside each role. "
                                 "Available keys: box [left,top,width,height] in whole poster percent, font, align, "
                                 "color, size (integer percent), max_lines, first_line (exact words for line 1, or null "
                                 "to clear a forced break). For headline only, depth can be behind_subject or null. "
                                 "Use it only for a small intentional overlap that leaves the words readable. "
                                 "Example: {\"changes\":{\"headline\":{\"size\":6,"
                                 "\"first_line\":\"<exact first line>\"}}}."},
    },
    "edit_region": {
        "findings": ("distraction", "scene"),
        "guide": "Repair an exact area of the scenery. Use for a stray prop, mark, or local texture defect. "
                 "Keep the selection away from the product silhouette. Everything outside will be preserved.",
        "arguments": {"box": "[left,top,width,height] in whole poster percent, inside the photo",
                      "instruction": "Name the unwanted detail and the scenery that should fill its place."},
    },
    "revise_scene": {
        "findings": ("scene", "product", "layout", "distraction"),
        "guide": "Rebuild the scene if the pose, product, light, or space for type fails across a large area. "
                 "Name a concrete camera, light, or staging change and where its effect should appear. "
                 "When a small detail is the selling point, consider a close crop of the actual detail and say "
                 "which other parts must remain identifiable. The product facts and exact copy stay fixed.",
        "arguments": {"instruction": "The visible problem and the specific photographic change to try."},
    },
    "select": {
        "findings": ("ready",),
        "guide": "Select the strongest visible version when the product and message read together. "
                 "Select an older version when the change made the poster worse.",
        "arguments": {},
    },
    "hold": {
        "findings": (),
        "guide": "Keep the available candidate for review when the visible issue cannot be fixed with the offered actions.",
        "arguments": {},
    },
}


def find_actions(finding, available):
    return {action: {"guide": guide["guide"], "arguments": guide["arguments"]}
            for action, guide in ACTION_GUIDES.items()
            if action in available and (action == "hold" or finding in guide["findings"])}
