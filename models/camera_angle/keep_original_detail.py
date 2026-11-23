"""다시 찍은 사진에서 원본이 보이는 영역은 원본 고해상도 픽셀로 덮습니다.

360도 배경은 해상도가 제한되어 원본이 흐려집니다. 카메라가 제자리에서 돌기만 했다면
원본 픽셀이 새 화면의 어디로 가는지 정확히 계산할 수 있으므로, 그 영역은 원본에서 직접 가져옵니다.
"""

import numpy as np
from PIL import Image, ImageFilter

from .camera_math import (CameraView, pixel_directions_for_photo, read_colors_from_photo, to_premultiplied_rgba,
                          to_uint8_image)
from .place_photo_in_360_view import ROWS_PER_STRIP, photo_camera


def paste_original_where_visible(result: Image.Image, photo: Image.Image, photo_fov_deg: float,
                                 camera: CameraView, feather_px: int) -> Image.Image:
    source = to_premultiplied_rgba(photo)
    original = photo_camera(photo, photo_fov_deg)
    colors = np.zeros((camera.height, camera.width, 4), dtype=np.float32)
    for start in range(0, camera.height, ROWS_PER_STRIP):
        stop = min(start + ROWS_PER_STRIP, camera.height)
        colors[start:stop] = read_colors_from_photo(source, pixel_directions_for_photo(camera, start, stop), original)
    alpha = colors[..., 3]
    straight = colors[..., :3] / np.maximum(alpha, 1e-6)[..., None]
    # 경계 안쪽으로 줄인 뒤 흐리게 해서, 원본과 생성 영역이 이어지는 선이 보이지 않게 합니다
    mask = to_uint8_image(alpha)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.MinFilter(2 * feather_px + 1)).filter(ImageFilter.GaussianBlur(feather_px / 2))
    weight = np.asarray(mask, dtype=np.float32)[..., None] / 255
    base = np.asarray(result.convert("RGB"), dtype=np.float32) / 255
    return to_uint8_image(straight * weight + base * (1 - weight))
