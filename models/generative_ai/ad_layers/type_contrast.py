"""실제 글자 획 뒤의 배경에 맞춰 색의 명도를 조정합니다."""

import numpy as np
from PIL import Image, ImageColor

from ..photo_retouch.adjustments import luminance, srgb_to_linear
from .layers import compose, to_float


def ink_contrast(lightness, background):
    return np.quantile(np.maximum(lightness, background) / np.minimum(lightness, background), 0.1)


def ink_background(backdrop, text, ink):
    x, y, w, h = (text[key] for key in ("x", "y", "w", "h"))
    pixels = to_float(backdrop.crop((x, y, x + w, y + h)))[ink]
    return luminance(srgb_to_linear(pixels)) + 0.05


def color_lightness(rgb):
    return luminance(srgb_to_linear(np.asarray(rgb) / 255)) + 0.05


def legible_tone(color, background, minimum):
    original = np.asarray(ImageColor.getrgb(color))
    if not background.size or ink_contrast(color_lightness(original), background) >= minimum:
        return color
    for step in range(1, 21):
        for neutral in (255, 0):
            shade = tuple(np.rint(original + (neutral - original) * step / 20).astype(int))
            if ink_contrast(color_lightness(shade), background) >= minimum:
                return "#%02X%02X%02X" % shade
    return None


def recolor_ink(path, alpha, color):
    bitmap = Image.new("RGBA", alpha.size, color)
    bitmap.putalpha(alpha)
    bitmap.save(path)


def readable_headline_colors(design, layout, backdrop, folder, foreground, unresolved=None, *, prefer_shared_tone=False):
    title = next(text for text in layout["texts"] if text["role"] == "headline")
    style = design["styles"]["headline"]
    original = title.get("lines", [title])
    preview_scale = min(1, 360 / backdrop.width, 640 / backdrop.height)
    accent = title.get("emphasis")
    accent_background, accent_minimum = [], 3
    samples = []
    for line in original:
        with Image.open(folder / line["file"]) as bitmap:
            pixels = np.array(bitmap.convert("RGBA"))
            bounds = bitmap.getchannel("A").getbbox()
        ink = pixels[..., 3] >= 128
        if line.get("depth") == "behind_subject":
            x, y = line["x"], line["y"]
            ink &= ~foreground[y:y + line["h"], x:x + line["w"]]
        large = bounds and line["max_lines"] == 1 and (bounds[3] - bounds[1]) * preview_scale >= 18
        minimum = 3 if large else 4.5
        emphasis = line.get("emphasis")
        base = line["color"]
        marked = np.zeros(ink.shape, dtype=bool)
        if emphasis:
            if emphasis["phrase"] == line["text"] or emphasis["color"].lower() == base.lower():
                marked[:] = True
            else:
                rgb = pixels[..., :3].astype(np.int32)
                base_error = ((rgb - ImageColor.getrgb(base)) ** 2).sum(axis=2)
                accent_error = ((rgb - ImageColor.getrgb(emphasis["color"])) ** 2).sum(axis=2)
                marked = accent_error < base_error
            background = ink_background(backdrop, line, ink & marked)
            if background.size:
                accent_background.append(background)
                accent_minimum = max(accent_minimum, minimum)
        color = legible_tone(base, ink_background(backdrop, line, ink & ~marked), minimum)
        if color is None and unresolved is not None:
            unresolved.append(f"제목 ‘{line['text']}’")
        samples.append((line, pixels, marked, color or base))

    accent_color = None
    if accent and accent_background:
        background = np.concatenate(accent_background)
        accent_color = legible_tone(accent["color"], background, accent_minimum)
        if prefer_shared_tone and accent_color != accent["color"]:
            accent_color = legible_tone(style["color"], background, accent_minimum)
    if accent and accent_background and accent_color is None and unresolved is not None:
        unresolved.append(f"강조 문구 ‘{accent['phrase']}’")
    lines = []
    for line, pixels, marked, color in samples:
        changes = {}
        visible = pixels[..., 3] > 0
        if color != line["color"]:
            pixels[visible & ~marked, :3] = ImageColor.getrgb(color)
            changes["color"] = color
        if (emphasis := line.get("emphasis")) and accent_color and accent_color != emphasis["color"]:
            pixels[visible & marked, :3] = ImageColor.getrgb(accent_color)
            changes["emphasis"] = emphasis | {"color": accent_color}
            if marked.all():
                changes["color"] = accent_color
        if changes:
            Image.fromarray(pixels).save(folder / line["file"])
            line = line | changes
        lines.append(line)
    if lines == original:
        return design, layout
    if "lines" in title:
        title = title | {"lines": lines}
        style = style | {"lines": [row | {"color": line["color"]}
                                   for row, line in zip(style["lines"], lines, strict=True)]}
    else:
        title = lines[0]
        style = style | {"color": title["color"]}
    if accent_color is not None:
        title = title | {"emphasis": accent | {"color": accent_color}}
        style = style | {"emphasis": style["emphasis"] | {"color": accent_color}}
    return design | {"styles": design["styles"] | {"headline": style}}, layout | {
        "texts": [title if text["role"] == "headline" else text for text in layout["texts"]]}


def readable_offer_colors(design, layout, backdrop, folder, unresolved=None):
    roles = ("product_name", "price")
    headline = design["styles"]["headline"]
    palette = [headline["color"], *(line.get("color", headline["color"]) for line in headline.get("lines") or [])]
    offer = [text for text in layout["texts"] if text["role"] in roles]
    colors = list(dict.fromkeys([*(text["color"] for text in offer), *palette, "#000000", "#FFFFFF"]))
    lightness = {color: color_lightness(ImageColor.getrgb(color)) for color in colors}
    masks, backgrounds, contrasts = {}, {}, {}
    for text in offer:
        with Image.open(folder / text["file"]) as layer:
            alpha = layer.getchannel("A")
        background = ink_background(backdrop, text, np.asarray(alpha) >= 128)
        if background.size:
            role = text["role"]
            masks[role] = alpha
            backgrounds[role] = background
            contrasts[role] = {
                color: ink_contrast(value, background) for color, value in lightness.items()
            }
    minimum = 4.5
    low = {text["role"] for text in offer if text["role"] in contrasts
           and contrasts[text["role"]][text["color"]] < minimum}
    if not low:
        return design, layout

    common = next((color for color in colors if all(scores[color] >= minimum for scores in contrasts.values())), None)
    styles = design["styles"].copy()
    texts = []
    for text in layout["texts"]:
        role = text["role"]
        current = text["color"]
        if role in contrasts and (common is not None or role in low):
            scores = contrasts[role]
            color = common or next((color for color in palette if scores[color] >= minimum), None)
            color = color or legible_tone(current, backgrounds[role], minimum)
            if color is None:
                color = max((current, "#000000", "#FFFFFF"), key=scores.__getitem__)
                if unresolved is not None:
                    unresolved.append(f"‘{text['text']}’")
            if color != current:
                recolor_ink(folder / text["file"], masks[role], color)
                styles[role] = styles[role] | {"color": color}
                text = text | {"color": color}
        texts.append(text)
    return design | {"styles": styles}, layout | {"texts": texts}


def readable_text_colors(design, layout, folder, foreground, *, prefer_shared_tone=False):
    backdrop = compose(layout | {"texts": []}, folder)
    artwork = layout["artwork"]
    if foreground.shape != (backdrop.height, backdrop.width) or artwork["x"] or artwork["y"]:
        mask = Image.new("1", backdrop.size)
        mask.paste(Image.fromarray(foreground), (artwork["x"], artwork["y"]))
        foreground = np.asarray(mask)
    unresolved = []
    design, layout = readable_headline_colors(design, layout, backdrop, folder, foreground, unresolved,
                                              prefer_shared_tone=prefer_shared_tone)
    design, layout = readable_offer_colors(design, layout, backdrop, folder, unresolved)
    issue = None
    if unresolved:
        issue = (f"{', '.join(unresolved)} 뒤의 밝기가 고르지 않아 글자색을 바꿔도 대비가 부족합니다. "
                 "사진과 문구를 유지하고 문제가 있는 글자 묶음을 밝기가 고른 곳으로 옮겨 주세요.")
    return design, layout, issue
