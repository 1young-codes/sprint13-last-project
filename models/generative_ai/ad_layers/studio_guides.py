"""포스터를 보고 찾을 수 있는 작업 가이드입니다."""

START_GUIDES = {
    "product_shoot": "Show the real product and one selling detail through light, scale, or a believable use scene.",
    "reference_poster": "Borrow the reference's composition or hierarchy. Use our product, visual identity and copy.",
    "new_concept": "Build one visual idea from the approved product benefit and requested placement.",
}

TEXT_CHANGES = ("Object keyed by headline, product_name or price. Put edits directly inside each role. "
                "Available keys: box [left,top,width,height] in whole poster percent, font, align, color, "
                "size (percent), max_lines, first_line (exact words for line 1, or null to clear a forced break). "
                "For headline only, depth can be behind_subject or null; keep a deliberate overlap small and legible. "
                "Headline emphasis is optional: {phrase: exact words, color: #RRGGBB}, or null to clear.")

ACTION_GUIDES = {
    "edit_text": {
        "findings": ("typography", "layout"),
        "guide": "Change only the typography that needs work. Keep the exact wording and photograph. "
                 "Use shared alignment and nearby spacing for related copy. Keep the heading in charge, "
                 "while giving the product name and price enough size and contrast to read together. "
                 "Move glyphs out of the product and check legibility in the actual image.",
        "arguments": {"changes": TEXT_CHANGES},
    },
    "edit_composition": {
        "findings": ("layout",),
        "guide": "When an arbitrary photo seam or an unused strip separates words and product, adjust the seam "
                 "and text positions together. Work with the existing photo, preserving its position, crop and subject. "
                 "Bring the headline toward the relevant feature through quiet space without covering it. Use the "
                 "actual visible bounds to place the name and price as a nearby, quieter group. Keep the exact copy. "
                 "Use a hard edge only when it supports an intentional graphic panel.",
        "arguments": {"instruction": "A concrete description of the broken reading path and how the words, edge "
                                      "and product should read together. Leave exact positions to the layout editor."},
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
    "change_viewpoint": {
        "findings": ("scene",),
        "guide": "Move the camera around the original product before rebuilding the photograph. Choose this when "
                 "a clearly different view will make a specific detail or arrangement easier to see. Name what the "
                 "new view needs to reveal, and keep the actual product recognizable.",
        "arguments": {"viewpoint": "One key from camera_viewpoints.",
                      "instruction": "Describe the visible detail to reveal and how the scene should support it."},
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
