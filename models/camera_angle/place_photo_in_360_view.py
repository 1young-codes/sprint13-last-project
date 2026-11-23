"""사진을 360도 배경 한가운데(정면)에 붙이고, 아직 비어 있는 곳의 마스크를 만듭니다."""

import numpy as np
from PIL import Image

from .camera_math import (CameraView, hex_to_rgb, pixel_directions_for_360_image, read_colors_from_photo,
                          to_premultiplied_rgba, to_uint8_image)

ROWS_PER_STRIP = 128  # 큰 배열을 한 번에 만들지 않도록 몇 행씩 나눠 계산합니다


def photo_camera(photo: Image.Image, photo_fov_deg: float) -> CameraView:
    """원본 사진을 찍은 카메라: 정면을 보고, 기울지 않았다고 둡니다."""
    return CameraView(width=photo.width, height=photo.height, fov_deg=photo_fov_deg)


def place_photo_in_360_view(photo: Image.Image, photo_fov_deg: float, view_width: int,
                            empty_color: str) -> tuple[Image.Image, Image.Image]:
    """(360도 배경 RGB, 빈 곳 마스크 L). 마스크는 흰색(255)이 비어 있어 채워야 할 곳입니다."""
    view_height = view_width // 2
    source = to_premultiplied_rgba(photo)
    camera = photo_camera(photo, photo_fov_deg)
    background = hex_to_rgb(empty_color)
    color = np.empty((view_height, view_width, 3), dtype=np.float32)
    empty = np.empty((view_height, view_width), dtype=np.float32)
    for start in range(0, view_height, ROWS_PER_STRIP):
        stop = min(start + ROWS_PER_STRIP, view_height)
        directions = pixel_directions_for_360_image(view_width, view_height, start, stop)
        placed = read_colors_from_photo(source, directions, camera)
        alpha = placed[..., 3:4]
        color[start:stop] = placed[..., :3] + background * (1 - alpha)
        empty[start:stop] = 1 - alpha[..., 0]
    return to_uint8_image(color), to_uint8_image(empty)
