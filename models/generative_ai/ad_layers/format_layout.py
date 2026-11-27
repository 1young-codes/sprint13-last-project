"""실제 글자 크기와 상품 영역으로 게시 비율에서 배치할 자리를 찾습니다."""

from math import ceil, sqrt

import numpy as np
from PIL import Image

from .layers import paste_clipped
from .scene_ad import pixel_box
from .text_render import measure_texts, minimum_font_size


def free_position(integral, size, extent, near, *, x_limits=(0, 100), y_limits=(0, 100), avoid=None):
    width, height = size
    w, h = extent
    x = np.arange(max(3, x_limits[0]), min(97, x_limits[1]) - w + 1)
    y = np.arange(max(3, y_limits[0]), min(97, y_limits[1]) - h + 1)
    if not x.size or not y.size:
        return None
    left, right = [np.rint(edge * width / 100).astype(int) for edge in (x, x + w)]
    top, bottom = [np.rint(edge * height / 100).astype(int) for edge in (y, y + h)]
    occupied = (integral[np.ix_(bottom, right)] - integral[np.ix_(top, right)]
                - integral[np.ix_(bottom, left)] + integral[np.ix_(top, left)])
    fits = occupied == 0
    if avoid is not None:
        ax, ay, aw, ah = avoid
        clear_x = (x >= ax + aw) | (x + w <= ax)
        clear_y = (y >= ay + ah) | (y + h <= ay)
        fits &= clear_x[None, :] | clear_y[:, None]
    rows, columns = np.nonzero(fits)
    if not rows.size:
        return None
    px, py = x[columns], y[rows]
    order = np.argmin(((px - near[0]) * width)**2 + ((py - near[1]) * height)**2)
    return int(px[order]), int(py[order])


def format_measurements(layout, size):
    width, height = size
    scale = width / layout["size"][0]
    heading_scale = min(scale, sqrt(scale * height / layout["size"][1]))
    tokens = [row for text in layout["texts"] for row in text.get("lines", (text,))]
    probes = []
    for text in tokens:
        ratio = heading_scale if text["role"].startswith("headline") else scale
        pixels = min(int(width * 0.2), max(minimum_font_size(width), round(text["max_font_size"] * ratio)))
        probes.append(text | {"x": 0, "y": 0, "w": width, "h": height, "max_font_size": pixels})
    return measure_texts(probes, size), {text["role"]: max(2, 100 * text["max_font_size"] / width) for text in probes}


def position_options(layout, size, foreground, product_bounds, measurement):
    width, height = size
    dimensions, font_sizes = measurement
    layers = {text["role"]: text for text in layout["texts"]}
    title = layers["headline"]
    rows = title.get("lines", [title])
    extents = {role: (ceil(100 * (w + 3) / width), ceil(100 * (h + 3) / height))
               for role, (w, h) in dimensions.items()}
    row_gap = max(1, round(width / height))
    group_gap = max(2, round(2 * width / height))
    header_w = max(extents[row["role"]][0] for row in rows)
    header_h = sum(extents[row["role"]][1] for row in rows) + (len(rows) - 1) * row_gap
    name_w, name_h = extents["product_name"]
    price_w, price_h = extents["price"]
    offer_w, offer_h = name_w + price_w + 1, max(name_h, price_h)
    stacked_offer = offer_w > 94
    if stacked_offer:
        offer_w, offer_h = max(name_w, price_w), name_h + row_gap + price_h
    integral = np.pad(np.cumsum(np.cumsum(foreground, axis=0, dtype=np.int32), axis=1), ((1, 0), (1, 0)))
    left, top, right, bottom = product_bounds
    center = (left + right) / 2
    upper_half = (top + bottom) // 2
    anchor = layout["design"]["boxes"]["headline"][0]
    if title["align"] == "center":
        anchor = center - header_w / 2
    elif title["align"] == "right":
        anchor = right - header_w

    options = {}

    def add(name, heading, offer):
        if not heading or not offer:
            return
        x, y = heading
        ox, oy = offer
        if stacked_offer:
            px, py = ox, oy + name_h + row_gap
        else:
            px, py = ox + name_w + 1, oy + (offer_h - price_h) // 2
            oy += (offer_h - name_h) // 2
        box = {"headline": [x, y, header_w, header_h], "product_name": [ox, oy, name_w, name_h],
               "price": [px, py, price_w, price_h]}
        line_boxes = []
        for line in rows:
            w, h = extents[line["role"]]
            offset = {"left": 0, "center": (header_w - w) // 2, "right": header_w - w}[line["align"]]
            line_boxes.append([x + offset, y, w, h])
            y += h + row_gap
        options[name] = {"boxes": box, "headline_lines": line_boxes if len(rows) > 1 else [], "font_sizes": font_sizes}

    heading = free_position(integral, size, (header_w, header_h), (anchor, top - header_h - group_gap),
                            y_limits=(0, upper_half))
    offer_anchor = center - offer_w / 2 if title["align"] == "center" else anchor
    offer = free_position(integral, size, (offer_w, offer_h), (offer_anchor, bottom + group_gap),
                          y_limits=(bottom + row_gap, 100))
    add("headline_above_offer_below", heading, offer)

    if heading:
        x, y = heading
        nearby = free_position(integral, size, (offer_w, offer_h), (x, y + header_h + group_gap),
                               y_limits=(y, upper_half), avoid=(x, y, header_w, header_h))
        add("headline_with_nearby_offer", heading, nearby)

    column = (max(header_w, offer_w), header_h + group_gap + offer_h)
    for name, x_limits, y_limits, near in (
        ("above", (0, 100), (0, upper_half), (anchor, top - column[1] - group_gap)),
        ("left", (0, left - row_gap), (0, 100), (left - column[0] - group_gap, (top + bottom - column[1]) / 2)),
        ("right", (right + row_gap, 100), (0, 100), (right + group_gap, (top + bottom - column[1]) / 2)),
    ):
        found = free_position(integral, size, column, near, x_limits=x_limits, y_limits=y_limits)
        if found:
            x, y = found
            horizontal_offset = (column[0] - header_w) // 2 if title["align"] == "center" else 0
            offer_offset = (column[0] - offer_w) // 2 if title["align"] == "center" else 0
            add(f"copy_{name}", (x + horizontal_offset, y), (x + offer_offset, y + header_h + group_gap))

    return options


def preview_placement(frame, layout, source, plan):
    canvas = frame.convert("RGBA")
    for text in layout["texts"]:
        boxes = plan["headline_lines"] if "lines" in text else [plan["boxes"][text["role"]]]
        for row, box in zip(text.get("lines", (text,)), boxes, strict=True):
            with Image.open(source / row["file"]) as image:
                bounds = image.getchannel("A").getbbox()
                if not bounds:
                    continue
                ink = image.crop(bounds).convert("RGBA")
            pixels = frame.width * plan["font_sizes"][row["role"]] / 100
            scale = pixels / row["max_font_size"]
            ink = ink.resize(tuple(max(1, round(side * scale)) for side in ink.size), Image.Resampling.LANCZOS)
            x, y, w, h = pixel_box(box, frame.size)
            offset = {"left": 0, "center": (w - ink.width) // 2, "right": w - ink.width}[row["align"]]
            paste_clipped(canvas, ink, x + offset, y + (h - ink.height) // 2)
    return canvas.convert("RGB")


def outer_margins(product_bounds, text_bounds):
    bounds = [product_bounds, *text_bounds]
    return {"top": min(box[1] for box in bounds), "bottom": 100 - max(box[3] for box in bounds),
            "left": min(box[0] for box in bounds), "right": 100 - max(box[2] for box in bounds)}
