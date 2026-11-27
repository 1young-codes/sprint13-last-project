import json

import pytest
from PIL import Image, ImageDraw
import numpy as np

from models.generative_ai.ad_layers.layers import to_float
from models.generative_ai.ad_layers.staging import move_source_piece, source_parts, stage_group, stage_subject


def test_staging_moves_source_pixels_without_copying_them():
    source = Image.new("RGBA", (120, 120))
    draw = ImageDraw.Draw(source)
    draw.ellipse((17, 18, 39, 42), fill="#af493b")
    draw.rectangle((48, 12, 88, 43), fill="#3b7399")
    staged, remaining, piece, _, _ = move_source_piece(source, source_parts(source.getchannel("A")), {
        "source_point": [23, 24], "target_base": [62, 80],
    })
    assert piece.tobytes() == source.crop((17, 18, 40, 43)).tobytes()
    assert staged.getpixel((27, 30))[3] == 0
    assert staged.getpixel((74, 84)) == source.getpixel((27, 30))
    assert staged.getpixel((74, 96)) == source.getpixel((28, 42))
    assert staged.getpixel((65, 25)) == remaining.getpixel((65, 25)) == source.getpixel((65, 25))


def test_staging_ignores_connected_shapes_and_mask_flecks():
    mask = Image.new("1", (100, 100))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((15, 15, 40, 40), fill=1)
    draw.rectangle((41, 41, 65, 65), fill=1)
    draw.point((81, 72), fill=1)
    assert source_parts(mask) is None


def test_staging_holds_a_move_that_fails_the_visual_check():
    photo = Image.new("RGB", (100, 100), "#af493b")
    mask = Image.new("1", photo.size)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((14, 20, 29, 50), fill=1)
    draw.rectangle((47, 21, 65, 51), fill=1)
    attempts = []

    class Reviewer:
        def ask(self, step, images, prompt, *, model):
            if step == "ad_staging":
                attempts.append(prompt)
                return json.dumps({"source_point": [20, 30], "target_base": [20, 65], "reason": "앞으로 이동"})
            assert step == "ad_staging_check" and len(images) == 5
            return '{"ready": false, "observation": "앞쪽에 놓이지 않았습니다"}'

    with pytest.raises(ValueError, match="앞쪽에 놓이지 않았습니다"):
        stage_subject(Reviewer(), photo, mask, source_parts(mask), "가까이에 놓습니다")
    assert len(attempts) == 2 and "앞쪽에 놓이지 않았습니다" in attempts[1]


def test_group_staging_places_every_source_component_once_without_resizing():
    photo = Image.new("RGB", (130, 85), "white")
    mask = Image.new("1", photo.size)
    colors = [(160, 30, 70), (230, 110, 20), (30, 120, 90), (25, 45, 110)]
    for left, color in zip((9, 37, 65, 93), colors, strict=True):
        box = left, 24, left + 18, 55
        photo.paste(color, box)
        mask.paste(1, box)
    calls = []

    class Reviewer:
        def ask(self, step, images, prompt, *, model):
            calls.append(step)
            data = json.loads(prompt.split("Input JSON follows:\n", 1)[1])
            if step == "ad_format_group_hero":
                assert data["pieces"] == [1, 2, 3, 4]
                return '{"hero":3,"reason":"세 번째 상품에 강조점이 있습니다"}'
            assert data["option_order"] == ["columns_1", "columns_2", "columns_3"]
            assert all(image.shape[:2] == (180, 100) for image in images[2:])
            return '{"choice":"columns_2","reason":"세 번째 상품이 앞쪽에서 분명히 보입니다"}'

    result, _ = stage_group(Reviewer(), photo, source_parts(mask), (100, 180), to_float(photo), "가로 배열", {})
    assert result.height > result.width
    pixels = np.asarray(result)
    for color in colors:
        assert (pixels == (*color, 255)).all(axis=2).sum() == 18 * 31
    assert (pixels[-1] == (*colors[2], 255)).all(axis=1).any()
    assert calls == ["ad_format_group_hero", "ad_format_group_check"]
