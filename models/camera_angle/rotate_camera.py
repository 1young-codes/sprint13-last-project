"""원본 사진을 원하는 방향, 화각, 렌즈의 카메라로 다시 찍습니다.

카메라가 제자리에서 돌기만 하므로, 새 화면의 각 픽셀이 원본 사진의 어디에서 오는지 정확히 계산할 수 있습니다.
원본이 찍지 않은 곳은 빈 곳으로 표시하고, 다음 단계(fill_empty_area.py)에서 채웁니다.
"""

import numpy as np
from PIL import Image

from .camera_math import (CameraView, hex_to_rgb, pixel_directions_for_photo, read_colors_from_photo,
                          to_premultiplied_rgba, to_uint8_image)

ROWS_PER_STRIP = 128  # 큰 배열을 한 번에 만들지 않도록 몇 행씩 나눠 계산합니다


def photo_camera(photo: Image.Image, photo_fov_deg: float) -> CameraView:
    """원본 사진을 찍은 카메라: 정면을 보고, 기울지 않았다고 둡니다."""
    return CameraView(width=photo.width, height=photo.height, fov_deg=photo_fov_deg)


def render_rotated_view(photo: Image.Image, photo_fov_deg: float, camera: CameraView,
                        empty_color: str) -> tuple[Image.Image, Image.Image]:
    """(새 카메라로 본 RGB, 빈 곳 마스크 L). 빈 곳은 empty_color로 칠하고, 마스크에서 흰색(255)입니다.
    어안 렌즈 원 바깥은 렌즈가 보지 않는 곳이라 검게 두고 빈 곳으로 치지 않습니다."""
    source = to_premultiplied_rgba(photo)
    original = photo_camera(photo, photo_fov_deg)
    background = hex_to_rgb(empty_color)
    color = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
    empty = np.zeros((camera.height, camera.width), dtype=np.float32)
    for start in range(0, camera.height, ROWS_PER_STRIP):
        stop = min(start + ROWS_PER_STRIP, camera.height)
        directions = pixel_directions_for_photo(camera, start, stop)
        seen = read_colors_from_photo(source, directions, original)
        alpha = seen[..., 3:4]
        lens = directions.any(axis=-1)[..., None]
        color[start:stop] = (seen[..., :3] + background * (1 - alpha)) * lens
        empty[start:stop] = (1 - alpha[..., 0]) * lens[..., 0]
    return to_uint8_image(color), to_uint8_image(empty)
