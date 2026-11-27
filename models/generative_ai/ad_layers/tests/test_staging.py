import json

import pytest
from PIL import Image, ImageDraw

from models.generative_ai.ad_layers.staging import move_source_piece, source_parts, stage_subject


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
