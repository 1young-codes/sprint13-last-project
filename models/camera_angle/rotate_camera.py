"""완성된 360도 배경에서 원하는 방향, 화각, 렌즈로 사진을 다시 찍습니다."""

import numpy as np
from PIL import Image

from .camera_math import CameraView, pixel_directions_for_photo, read_colors_from_360_image, to_uint8_image
from .place_photo_in_360_view import ROWS_PER_STRIP


def render_rotated_view(view_360: Image.Image, camera: CameraView) -> Image.Image:
    source = np.asarray(view_360.convert("RGB"), dtype=np.float32) / 255
    result = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
    for start in range(0, camera.height, ROWS_PER_STRIP):
        stop = min(start + ROWS_PER_STRIP, camera.height)
        directions = pixel_directions_for_photo(camera, start, stop)
        colors = read_colors_from_360_image(source, directions)
        colors[~directions.any(axis=-1)] = 0  # 어안 렌즈 원 바깥은 검게
        result[start:stop] = colors
    return to_uint8_image(result)
