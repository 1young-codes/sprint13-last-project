"""깊이 추정 결과(점 구름)를 이용해 카메라를 피사체 중심으로 돌려 새 시점을 그립니다.

카메라를 옮기면 원본에서 가려져 있던 곳이 드러납니다. 그 부분은 비워 두고 구멍 마스크로 돌려주며,
구멍은 complete_new_viewpoint.py가 생성 모델로 채웁니다.
"""

import math

import numpy as np
from PIL import Image

from .camera_math import hex_to_rgb, rotation_matrix_from_angles, to_uint8_image


def find_subject_center(points: np.ndarray, valid: np.ndarray, center_crop: float) -> np.ndarray:
    """화면 가운데 center_crop 비율 영역에 있는 점들의 중앙값을 피사체 중심으로 봅니다."""
    h, w = valid.shape
    top, left = int(h * (1 - center_crop) / 2), int(w * (1 - center_crop) / 2)
    crop = valid[top:h - top, left:w - left]
    if not crop.any():
        raise ValueError("화면 가운데에서 깊이를 추정한 점을 찾지 못했습니다")
    return np.median(points[top:h - top, left:w - left][crop], axis=0)


def orbit_camera_pose(subject_center: np.ndarray, orbit_yaw_deg: float,
                      orbit_pitch_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(회전 행렬, 카메라 위치). 원래 카메라는 원점에서 +z를 봅니다.
    카메라를 피사체 중심을 축으로 돌리므로, 피사체 중심은 새 화면에서도 같은 자리에 남습니다."""
    rotation = rotation_matrix_from_angles(orbit_yaw_deg, orbit_pitch_deg, 0)
    position = subject_center - rotation @ subject_center
    return rotation, position


def render_from_moved_camera(photo: Image.Image, points: np.ndarray, valid: np.ndarray, fov_deg: float,
                             orbit_yaw_deg: float, orbit_pitch_deg: float, center_crop: float,
                             hole_color: str) -> tuple[Image.Image, Image.Image]:
    """(새 시점 RGB, 구멍 마스크 L). 결과 크기는 원본과 같고, 마스크는 흰색(255)이 비어 있는 곳입니다.

    points: (H, W, 3) 카메라 기준 좌표, x 오른쪽, y 위, z 앞. valid: (H, W) 믿을 수 있는 점.
    """
    # ponytail: 점 하나를 픽셀 하나에 찍는 방식. 카메라가 피사체에 다가가면 작은 틈이 생기며, 품질이 부족하면 삼각형 메시 렌더링으로 바꾼다.
    h, w = valid.shape
    colors = np.asarray(photo.convert("RGB").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    rotation, position = orbit_camera_pose(find_subject_center(points, valid, center_crop),
                                           orbit_yaw_deg, orbit_pitch_deg)
    local = (points[valid] - position) @ rotation  # 각 점을 새 카메라 기준 좌표로
    tangent = math.tan(math.radians(fov_deg) / 2)
    z = local[:, 2]
    front = z > 1e-6
    x = (local[front, 0] / z[front] / tangent + 1) / 2 * w
    y = (-local[front, 1] / z[front] / (tangent * h / w) + 1) / 2 * h
    column, row = np.floor(x).astype(int), np.floor(y).astype(int)
    inside = (column >= 0) & (column < w) & (row >= 0) & (row < h)
    pixel = row[inside] * w + column[inside]
    depth = z[front][inside]
    source = colors[valid][front][inside]
    # 한 픽셀에 여러 점이 떨어지면 가장 가까운 점을 씁니다
    order = np.lexsort((depth, pixel))
    first = np.unique(pixel[order], return_index=True)[1]
    nearest = order[first]
    image = np.broadcast_to(hex_to_rgb(hole_color), (h * w, 3)).copy()
    image[pixel[nearest]] = source[nearest]
    hole = np.ones(h * w, dtype=np.float32)
    hole[pixel[nearest]] = 0
    return to_uint8_image(image.reshape(h, w, 3)), to_uint8_image(hole.reshape(h, w))
