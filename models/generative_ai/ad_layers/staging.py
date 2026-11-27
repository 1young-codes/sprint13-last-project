"""원본 상품의 분리 가능한 요소를 옮겨 촬영 구도를 만듭니다."""

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import find_objects, label

from .creative_brief import ask_stage, required_text
from .layers import to_float


PLAN = """The first image is the source photo. The second shows the isolated product at identical coordinates,
with a grid marked every 10 percent. Gray is transparent. Use the grid to locate the actual parts.
Move one physically separable piece of the actual subject to create the requested staging. Keep its size,
appearance, product count, and all other parts. Select an actual, movable piece, never a printed design or a
material attached to the object. Pick a position that supports the idea and is physically believable once lit.
Locate where its bottom edge will touch the visible surface using the grid and original photo. For a move
forward on a surface seen from above, that contact should be lower in the image than the pieces behind it.
Use coordinates in whole percentages of the original image. Return source_point [x, y] well inside a piece
whose visible silhouette does not touch other pieces, target_base [x, y] for its new bottom-center contact
point, and reason (Korean). Do not resize the piece."""

CHECK = """The first image is the original photo. The second is the real staging guide made by moving an isolated
piece. Gray is transparent. Images three and four zoom in on the old location before and after the move,
respectively. The last image shows the piece by itself. First inspect the piece: it must show the full visible
silhouette, including small protrusions, without cut edges through its interior or pixels cut from a neighbor.
Compare the two crops. Reject colored slivers or outlines from the moved piece left at the old location.
Empty background or a surface previously hidden by the object is expected. In the full staged image, the
other product pieces must still be recognizable. For a move forward, compare
the bottom contact points in the second image. A sideways move alone does not establish foreground.
Reject a move that does not support the idea. Return ready (boolean) and observation (one concrete Korean sentence)."""

GROUP_HERO = """Image 1 is the original product photo, image 2 is the approved poster, and image 3
labels source pieces. The new format needs a different arrangement of the same pieces. Identify the one
that carries the approved campaign's visual action or focus, so it can lead the new arrangement.
Return hero, a number from pieces or null if no individual piece should lead, and reason in Korean."""

GROUP_CHECK = """Image 1 is the original product photo and image 2 is the approved campaign.
The rest are actual cutout arrangements of the original pieces, in option_order, on the target canvas.
Choose the arrangement that best uses the long axis while leaving a connected space for this exact copy.
Read the staggered rows as depth on one shared surface from the existing camera, with the leading piece
readable. Gray is a placeholder for the setting. Reject a cramped grid of unrelated objects, components
that cannot realistically be moved, and layouts that merely repeat the original line of pieces.
Return choice as a key from option_order or null, and one visible reason in Korean."""

STAGED_IDENTITY = """Compare the moved item in three images: the real source item isolated on gray, the newly
photographed piece and its surroundings at close range, and the whole new photograph. Verify shape and
construction of the advertised item, especially the small working parts identified in product_detail. A different
angle, lighting, or physically expected traces of chosen use are allowed. Replacing a visible working part
with a different material or shape is a mismatch even when its overall use and main silhouette look plausible.
Return match (boolean) and observation (one concrete Korean sentence naming what the close images show)."""


def source_parts(mask):
    if mask is None:
        return None
    labels, _ = label(np.asarray(mask) > 0, structure=np.ones((3, 3), bool))
    counts = np.bincount(labels.ravel())
    # Tiny mask flecks are not separately movable product pieces.
    visible = np.flatnonzero(counts[1:] >= max(1, counts[1:].sum() * 0.02)) + 1
    if len(visible) < 2:
        return None
    return np.where(np.isin(labels, visible), labels, 0)


def arrange_rows(rows, gap, hero):
    widths = [sum(piece.width for piece in row) + gap * (len(row) - 1) for row in rows]
    heights = [max(piece.height for piece in row) for row in rows]
    width = max(widths)
    result = Image.new("RGBA", (width, sum(heights) + gap * (len(rows) - 1)))
    highlight = Image.new("RGBA", result.size) if hero is not None else None
    y = 0
    for index, (row, row_width, height) in enumerate(zip(rows, widths, heights, strict=True)):
        slack = width - row_width
        x = round(slack * (0.25 if index % 2 == 0 else 0.75)) if hero is not None else slack // 2
        for piece in row:
            position = x, y + height - piece.height
            result.alpha_composite(piece, position)
            if piece is hero:
                highlight.paste(piece, position)
            x += piece.width + gap
        y += height + gap
    box = result.getbbox()
    return result.crop(box), highlight.crop(box) if highlight is not None else None


def stage_group(ai, photo, parts, size, approved, issue, copy):
    items = []
    for index, bounds in enumerate(find_objects(parts), 1):
        if bounds is None:
            continue
        y, x = bounds
        box = x.start, y.start, x.stop, y.stop
        if x.start == 0 or y.start == 0 or x.stop == photo.width or y.stop == photo.height:
            return None
        piece = photo.crop(box).convert("RGBA")
        piece.putalpha(Image.fromarray(parts[bounds] == index).convert("L"))
        items.append((index, box, piece))
    if len(items) < 3:
        return None

    marked = photo.copy()
    draw = ImageDraw.Draw(marked)
    for index, box, _ in items:
        draw.rectangle(box, outline="white", width=max(1, min(photo.size) // 180))
        draw.text(box[:2], str(index), fill="white", stroke_width=2, stroke_fill="black")
    data = {"pieces": [index for index, _, _ in items], "target_canvas": size, "format_issue": issue, "copy": copy}
    images = [to_float(photo), approved]
    focus = ask_stage(ai, "ad_format_group_hero", GROUP_HERO, data, [*images, to_float(marked)])
    hero = focus.get("hero")
    if hero is not None and (type(hero) is not int or hero not in data["pieces"]):
        raise ValueError("상품군에서 강조할 원본 요소를 찾지 못했습니다")
    items.sort(key=lambda item: (item[0] == hero, item[1][0]))
    pieces = [piece for _, _, piece in items]
    gap = max(1, round(float(np.median([min(piece.size) for piece in pieces])) / 12))
    options = {}
    max_columns = min(len(pieces) - (hero is not None), 4)
    for columns in range(1, max_columns + 1):
        others = pieces[:-1] if hero is not None and columns > 1 else pieces
        rows = [others[i:i + columns] for i in range(0, len(others), columns)]
        if len(others) != len(pieces):
            rows.append(pieces[-1:])
        if len(rows) == 1 and size[1] > size[0]:
            continue
        options[f"columns_{columns}"] = arrange_rows(rows, gap, pieces[-1] if hero is not None else None)
    previews = []
    for group, _ in options.values():
        canvas = Image.new("RGB", size, "#808080")
        fitted = ImageOps.contain(group, (round(size[0] * 0.88), round(size[1] * 0.70)), Image.Resampling.LANCZOS)
        canvas.paste(fitted, ((size[0] - fitted.width) // 2, size[1] * 4 // 5 - fitted.height),
                     fitted.getchannel("A"))
        previews.append(to_float(canvas))
    answer = ask_stage(ai, "ad_format_group_check", GROUP_CHECK,
                       data | {"option_order": list(options), "hero": hero}, [*images, *previews])
    choice = answer.get("choice")
    if choice is None:
        return None
    if not isinstance(choice, str) or choice not in options:
        raise ValueError("게시 규격에 맞춰 상품군을 놓을 방식을 확인하지 못했습니다")
    group, highlight = options[choice]
    return group, highlight, required_text(answer, "reason", "상품군 재배치 검토")


def padded_box(box, margin, size):
    x0, y0, x1, y1 = box
    return max(0, x0 - margin), max(0, y0 - margin), min(size[0], x1 + margin), min(size[1], y1 + margin)


def staged_evidence(folder, layout, scene):
    revision = layout.get("revision", {})
    box = revision.get("staged_piece_box")
    if (not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box)
            or not (0 <= box[0] < box[2] <= scene.width and 0 <= box[1] < box[3] <= scene.height)):
        raise ValueError("옮긴 상품을 확대할 영역을 확인하지 못했습니다")
    with Image.open(folder / "staged-piece.png") as image:
        original = image.convert("RGB")
    # A photographed group can shift relative to its multi-piece guide.
    margin = max(box[2] - box[0], box[3] - box[1]) // (2 if revision.get("staged_group") else 4)
    return original, padded_box(box, margin, scene.size)


def check_staged_identity(ai, folder, layout, scene):
    original, bounds = staged_evidence(folder, layout, scene)
    identity = ask_stage(ai, "studio_staged_identity", STAGED_IDENTITY, {
        "product_detail": layout.get("art_direction_plan", {}).get("evidence"),
        "staging_intent": layout.get("revision", {}).get("staging_intent"),
    }, [to_float(original), to_float(scene.crop(bounds)), to_float(scene)])
    if type(identity.get("match")) is not bool:
        raise ValueError("옮긴 상품의 원본 대조 결과를 확인하지 못했습니다")
    return identity["match"], required_text(identity, "observation", "옮긴 상품 검토")


def on_gray(subject):
    guide = Image.new("RGB", subject.size, "#808080")
    guide.paste(subject, mask=subject.getchannel("A"))
    return guide


def coordinate_guide(subject):
    guide = on_gray(subject)
    draw = ImageDraw.Draw(guide)
    for value in range(0, 100, 10):
        x, y = point_pixels([value, value], guide.size)
        draw.line((x, 0, x, guide.height), fill="#555555")
        draw.line((0, y, guide.width, y), fill="#555555")
        draw.text((x + 2, 3), str(value), fill="white", stroke_width=1, stroke_fill="black")
        draw.text((3, y + 17), str(value), fill="white", stroke_width=1, stroke_fill="black")
    return guide


def stage_subject(ai, photo, mask, parts, intent):
    source = photo.convert("RGBA")
    source.putalpha(mask.convert("L"))
    data = {"staging_intent": intent}
    source_images = [to_float(photo), to_float(coordinate_guide(source))]
    for _ in range(2):
        answer = ask_stage(ai, "ad_staging", PLAN, data, source_images)
        try:
            staged, remaining, piece, bounds, position = move_source_piece(source, parts, answer)
            proof = [photo, on_gray(staged), on_gray(source.crop(bounds)),
                     on_gray(remaining.crop(bounds)), on_gray(piece)]
            review = ask_stage(ai, "ad_staging_check", CHECK, data | {
                "move": required_text(answer, "reason", "원본 재배치"),
            }, list(map(to_float, proof)))
            observation = required_text(review, "observation", "원본 재배치 검토")
            if review.get("ready") is True:
                framed = staged.getbbox()
                moved = Image.new("RGBA", staged.size)
                moved.paste(piece, position)
                return staged.crop(framed), moved.crop(framed), observation
        except ValueError as error:
            observation = str(error)
        data.update(previous_selection=answer, rejection=observation)
    raise ValueError(f"원본 재배치를 다시 정해야 합니다. {observation}")


def point_pixels(point, size):
    if (not isinstance(point, list) or len(point) != 2
            or any(type(value) is not int or not 0 <= value < 100 for value in point)):
        raise ValueError("원본 재배치의 점은 0에서 99 사이 정수 두 개여야 합니다")
    return tuple(value * side // 100 for value, side in zip(point, size, strict=True))


def move_source_piece(source, parts, answer):
    px, py = point_pixels(answer.get("source_point"), source.size)
    tx, ty = point_pixels(answer.get("target_base"), source.size)
    if parts is None or not parts[py, px]:
        raise ValueError("주변 물체와 분리된 상품 요소 안에 기준점을 놓아야 합니다")
    selection = parts == parts[py, px]
    piece_mask = Image.fromarray(selection)
    bounds = piece_mask.getbbox()
    if bounds[0] == 0 or bounds[1] == 0 or bounds[2] == source.width or bounds[3] == source.height:
        raise ValueError("원본 사진의 가장자리에 잘린 요소는 옮길 수 없습니다")
    piece = source.copy()
    piece.putalpha(piece_mask.convert("L"))
    piece = piece.crop(bounds)
    position = tx - piece.width // 2, ty - piece.height + 1
    if position == bounds[:2]:
        raise ValueError("옮긴 위치가 원래 위치와 같습니다")
    if not (0 <= position[0] <= source.width - piece.width and 0 <= position[1] <= source.height - piece.height):
        raise ValueError("옮긴 요소가 사진 안에 들어오지 않습니다")
    remaining = source.copy()
    remaining.putalpha(Image.fromarray(np.where(selection, 0, np.asarray(source.getchannel("A")))))
    staged = remaining.copy()
    staged.alpha_composite(piece, position)
    evidence_box = (max(0, bounds[0] - piece.width // 5), max(0, bounds[1] - piece.height // 5),
                    min(source.width, bounds[2] + piece.width // 5), min(source.height, bounds[3] + piece.height // 5))
    return staged, remaining, piece, evidence_box, position
