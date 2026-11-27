"""원본 상품의 분리 가능한 한 요소를 옮긴 촬영 구도를 만듭니다."""

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

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
