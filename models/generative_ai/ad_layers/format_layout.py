"""실제 글자 크기와 상품 영역으로 게시 비율에서 배치할 자리를 찾습니다."""

from math import ceil, sqrt

import numpy as np
from PIL import Image

from .layers import paste_clipped
from .scene_ad import pixel_box
from .text_render import fitted_text_sizes, measure_texts, minimum_font_size


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


def offer_reflow_options(layout, design, occupied, product_bounds, ink_heights, target_height):
    width, height = size = tuple(layout["size"])
    roles = ("product_name", "price")
    offer = [text for text in layout["texts"] if text["role"] in roles]
    fitted = fitted_text_sizes(offer, size)
    probes = []
    for text in offer:
        role = text["role"]
        scale = max(1, target_height / ink_heights[role])
        pixels = min(int(width * 0.2), ceil(fitted[role] * scale))
        probes.append(text | {"w": width, "h": height, "max_lines": 1, "max_font_size": pixels})
    measured = measure_texts(probes, size)
    name_w, name_h = (ceil(100 * (n + 4) / side)
                      for n, side in zip(measured["product_name"], size, strict=True))
    price_w, price_h = (ceil(100 * (n + 4) / side)
                        for n, side in zip(measured["price"], size, strict=True))
    row_w, row_h = name_w + 2 + price_w, max(name_h, price_h)
    stack_w, stack_h = max(name_w, price_w), name_h + 1 + price_h
    left, top, right, bottom = product_bounds
    integral = np.pad(np.cumsum(np.cumsum(occupied, axis=0, dtype=np.int32), axis=1), ((1, 0), (1, 0)))
    title = design["boxes"]["headline"]
    styles = design["styles"] | {
        text["role"]: {key: value for key, value in design["styles"][text["role"]].items() if key != "break_after"} | {
            "max_lines": 1, "size": max(2, 100 * text["max_font_size"] / width)}
        for text in probes
    }

    options = {}
    nearby = {}
    for role, (w, h) in zip(roles, ((name_w, name_h), (price_w, price_h)), strict=True):
        x, y, previous_w, previous_h = design["boxes"][role]
        align = {"left": 0, "center": 0.5, "right": 1}[styles[role]["align"]]
        nearby[role] = [round(x + (previous_w - w) * align), round(y + (previous_h - h) / 2), w, h]
    if all(text["max_lines"] == 1 for text in offer):
        options["near_current"] = design | {"boxes": design["boxes"] | nearby, "styles": styles}

    styles = styles | {role: styles[role] | {"align": "left"} for role in roles}
    alignment = design["styles"]["headline"].get("align", "center")
    anchor = {"left": title[0], "center": (left + right - row_w) / 2,
              "right": title[0] + title[2] - row_w}[alignment]
    below = free_position(integral, size, (row_w, row_h), (anchor, bottom + 1),
                          y_limits=(round((top + 2 * bottom) / 3), 97), avoid=title)
    if below:
        x, y = below
        boxes = {"product_name": [x, y + (row_h - name_h) // 2, name_w, name_h],
                 "price": [x + name_w + 2, y + (row_h - price_h) // 2, price_w, price_h]}
        options["below"] = design | {"boxes": design["boxes"] | boxes, "styles": styles}

    center = round((left + right) / 2)
    for side, limits, near_x in (("left", (0, center), left - stack_w - 1),
                                 ("right", (center, 100), right + 1)):
        beside = free_position(integral, size, (stack_w, stack_h), (near_x, (top + bottom - stack_h) / 2),
                               x_limits=limits, y_limits=(top, min(bottom, 95)), avoid=title)
        if beside:
            x, y = beside
            boxes = {"product_name": [x, y, name_w, name_h],
                     "price": [x, y + name_h + 1, price_w, price_h]}
            options[side] = design | {"boxes": design["boxes"] | boxes, "styles": styles}
    return options


def supporting_price_size(focus, proposed, lead, minimum):
    if focus == "price" or lead is None:
        return proposed
    return min(proposed, max(minimum, 0.60 * lead))


def format_type_hierarchy(styles, focus):
    if not isinstance(styles, dict) or not isinstance(styles.get("price"), dict):
        return styles
    price = styles["price"].get("size")
    if type(price) not in (int, float) or not 2 <= price <= 20:
        return styles
    front = styles.get("product_name" if focus == "product_name" else "headline")
    if not isinstance(front, dict):
        return styles
    size = front.get("size")
    lines = front.get("lines", ())
    sizes = ([line.get("size", size) for line in lines if isinstance(line, dict)]
             if isinstance(lines, (list, tuple)) and lines else [size])
    lead = max((value for value in sizes if type(value) in (int, float) and 2 <= value <= 20), default=None)
    adjusted = supporting_price_size(focus, price, lead, 2)
    return styles if adjusted == price else styles | {"price": styles["price"] | {"size": adjusted}}


def format_measurements(layout, size, *, finished_photo=False):
    width, height = size
    scale = width / layout["size"][0]
    heading_scale = min(scale, sqrt(scale * height / layout["size"][1]))
    tokens = [row for text in layout["texts"] for row in text.get("lines", (text,))]
    if finished_photo:
        pixels_by_role = fitted_text_sizes(tokens, size)
    else:
        pixels_by_role = {}
        for text in tokens:
            ratio = heading_scale if text["role"].startswith("headline") else scale
            pixels = min(int(width * 0.2), max(minimum_font_size(width), round(text["max_font_size"] * ratio)))
            pixels_by_role[text["role"]] = pixels
    focus = layout.get("design", {}).get("focus")
    lead = (pixels_by_role.get("product_name") if focus == "product_name" else
            max((pixels for role, pixels in pixels_by_role.items() if role.startswith("headline")), default=None))
    pixels_by_role["price"] = round(supporting_price_size(focus, pixels_by_role["price"], lead, minimum_font_size(width)))
    probes = [text | {"x": 0, "y": 0, "w": width, "h": height, "max_font_size": pixels_by_role[text["role"]]}
              for text in tokens]
    return measure_texts(probes, size), {text["role"]: max(2, 100 * text["max_font_size"] / width) for text in probes}


def position_options(layout, size, foreground, product_bounds, measurement, *, finished_photo=False):
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
    approved_x, approved_y, approved_width, approved_height = layout["design"]["boxes"]["headline"]
    anchor = approved_x
    if title["align"] == "center":
        anchor = center - header_w / 2
    elif title["align"] == "right":
        anchor = right - header_w

    options = {}

    def add(name, heading, offer, *, stack=stacked_offer):
        if not heading or not offer:
            return
        x, y = heading
        ox, oy = offer
        if stack:
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
    group_alignment = {"left": 0, "center": 0.5, "right": 1}[title["align"]]
    approved_near = (
        approved_x + (approved_width - column[0]) * group_alignment,
        approved_y + (approved_height - header_h) / 2,
    )
    for name, x_limits, y_limits, near in (
        ("above", (0, 100), (0, upper_half), (anchor, top - column[1] - group_gap)),
        ("left", (0, left - row_gap), (0, 100), (left - column[0] - group_gap, (top + bottom - column[1]) / 2)),
        ("right", (right + row_gap, 100), (0, 100), (right + group_gap, (top + bottom - column[1]) / 2)),
        ("near_approved", (0, 100), (0, 100), approved_near),
    ):
        found = free_position(integral, size, column, near, x_limits=x_limits, y_limits=y_limits)
        if found:
            x, y = found
            horizontal_offset = int((column[0] - header_w) * group_alignment)
            offer_offset = int((column[0] - offer_w) * group_alignment)
            add(f"copy_{name}", (x + horizontal_offset, y), (x + offer_offset, y + header_h + group_gap))

    if finished_photo:
        mid = round(center)
        for name, limits, x in (("left", (0, mid), max(anchor, left - header_w - 1)),
                                ("right", (mid, 100), right)):
            beside = free_position(integral, size, (header_w, header_h),
                                   (x, top + 0.4 * (bottom - top) - header_h / 2), x_limits=limits,
                                   y_limits=(top, bottom))
            if beside:
                hx, hy = beside
                compact = (max(name_w, price_w), name_h + row_gap + price_h)
                supporting = free_position(integral, size, compact, (hx, hy + header_h + group_gap),
                                           y_limits=(hy + header_h, 100))
                add(f"headline_beside_{name}", beside, supporting, stack=True)

        below = free_position(integral, size, column, (center - column[0] / 2, bottom + group_gap),
                              y_limits=(bottom + row_gap, 100))
        if below:
            x, y = below
            heading_x = x + (column[0] - header_w) // 2 if title["align"] == "center" else x
            offer_x = x + (column[0] - offer_w) // 2 if title["align"] == "center" else x
            add("copy_below", (heading_x, y), (offer_x, y + header_h + group_gap))

    return options


def measured_styles(design, plan, colors=None):
    colors = colors if colors is not None else {}
    line_roles = [f"headline_line_{index}" for index, _ in enumerate(plan["headline_lines"], 1)]
    if not isinstance(colors, dict) or colors.keys() - (plan["boxes"].keys() | set(line_roles)):
        raise ValueError("게시 규격 문구의 색상을 확인하지 못했습니다")
    styles = {role: style.copy() for role, style in design["styles"].items()}
    for role in styles.keys() & plan["font_sizes"].keys():
        styles[role]["size"] = plan["font_sizes"][role]
    for role in colors.keys() & styles.keys():
        styles[role]["color"] = colors[role]
    if plan["headline_lines"]:
        headline = styles["headline"]
        rows = []
        for line, box, role in zip(headline["lines"], plan["headline_lines"], line_roles, strict=True):
            row = line | {"box": box, "size": plan["font_sizes"][role]}
            if role in colors or "headline" in colors:
                row["color"] = colors[role] if role in colors else colors["headline"]
            rows.append(row)
        headline["lines"] = rows
    return styles


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
