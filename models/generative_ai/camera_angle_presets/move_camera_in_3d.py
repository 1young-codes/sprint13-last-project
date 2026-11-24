"""깊이 추정 결과(점 구름)를 이용해 카메라를 피사체 중심으로 돌려 새 시점을 그립니다.

카메라를 옮기면 원본에서 가려져 있던 곳이 드러납니다. 그 부분은 비워 두고 구멍 마스크로 돌려주며,
구멍은 complete_new_viewpoint.py가 생성 모델로 채웁니다.
"""

import math
import warnings

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



def remove_depth_edge_points(points: np.ndarray, valid: np.ndarray, tolerance: float) -> np.ndarray:
    """물체 경계에서 깊이가 앞뒤로 튀는 점(떠다니는 점)을 뺀 valid.
    주변 3×3 픽셀의 깊이 차이가 깊이 × tolerance를 넘으면 경계로 봅니다 (ai-generation grid_surface와 같은 기준)."""
    depth = np.where(valid, points[..., 2], np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(depth, 1, mode="edge"), (3, 3))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 전부 비어 있는 창
        spread = np.nanmax(windows, axis=(-1, -2)) - np.nanmin(windows, axis=(-1, -2))
    return valid & (spread <= tolerance * np.maximum(depth, 1e-6))


def orbit_camera_pose(subject_center: np.ndarray, orbit_yaw_deg: float,
                      orbit_pitch_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(회전 행렬, 카메라 위치). 원래 카메라는 원점에서 +z를 봅니다.
    카메라를 피사체 중심을 축으로 돌리므로, 피사체 중심은 새 화면에서도 같은 자리에 남습니다."""
    rotation = rotation_matrix_from_angles(orbit_yaw_deg, orbit_pitch_deg, 0)
    position = subject_center - rotation @ subject_center
    return rotation, position


def render_from_moved_camera(photo: Image.Image, points: np.ndarray, valid: np.ndarray, fov_deg: float,
                             orbit_yaw_deg: float, orbit_pitch_deg: float, center_crop: float,
                             hole_color: str, point_size_px: int = 1,
                             gap_fill_ratio: float = 0.0, background_depth_share: float = 1.0) -> tuple[Image.Image, Image.Image]:
    """(새 시점 RGB, 구멍 마스크 L). 결과 크기는 원본과 같고, 마스크는 흰색(255)이 비어 있는 곳입니다.

    points: (H, W, 3) 카메라 기준 좌표, x 오른쪽, y 위, z 앞. valid: (H, W) 믿을 수 있는 점.
    point_size_px: 점 하나가 덮는 정사각형 크기. 카메라가 움직여 표면이 늘어나면 점 사이에 그물 모양 틈이 생기는데,
    생성 모델이 그 무늬를 물체로 착각하므로 점을 키워 틈을 메웁니다.
    gap_fill_ratio: 앞 물체 뒤에서 드러난 틈을 뒤쪽 배경으로 메울 최대 폭 (화면 가로 대비 비율). 0이면 메우지 않습니다.
    background_depth_share: 주변에서 가장 먼 깊이의 이 비율보다 먼 점만 뒤쪽 배경으로 봅니다.
    """
    # ponytail: 점을 정사각형으로 찍는 방식. 물체 가장자리가 point_size_px만큼 두꺼워지며, 부족하면 삼각형 메시 렌더링으로 바꾼다.
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
    offsets = range(-((point_size_px - 1) // 2), point_size_px // 2 + 1)
    column = np.concatenate([np.floor(x).astype(int) + dx for dy in offsets for dx in offsets])
    row = np.concatenate([np.floor(y).astype(int) + dy for dy in offsets for dx in offsets])
    copies = len(offsets) ** 2
    inside = (column >= 0) & (column < w) & (row >= 0) & (row < h)
    pixel = row[inside] * w + column[inside]
    depth = np.tile(z[front], copies)[inside]
    source = np.tile(colors[valid][front], (copies, 1))[inside]
    # 한 픽셀에 여러 점이 떨어지면 가장 가까운 점을 씁니다
    order = np.lexsort((depth, pixel))
    first = np.unique(pixel[order], return_index=True)[1]
    nearest = order[first]
    image = np.zeros((h * w, 3), dtype=np.float32)
    image[pixel[nearest]] = source[nearest]
    distance = np.zeros(h * w, dtype=np.float32)  # 0은 빈 곳
    distance[pixel[nearest]] = depth[nearest]
    image, distance = image.reshape(h, w, 3), distance.reshape(h, w)
    image, distance = fill_gaps_behind_objects(image, distance, round(gap_fill_ratio * w),
                                               background_depth_share)
    hole = distance == 0
    image[hole] = hex_to_rgb(hole_color)
    return to_uint8_image(image), to_uint8_image(hole.astype(np.float32))



def local_max(values: np.ndarray, radius: int) -> np.ndarray:
    """각 픽셀 주변 (2 × radius + 1) 정사각형 안의 최댓값."""
    window = 2 * radius + 1
    rows = np.lib.stride_tricks.sliding_window_view(np.pad(values, ((radius, radius), (0, 0))), window, axis=0).max(-1)
    return np.lib.stride_tricks.sliding_window_view(np.pad(rows, ((0, 0), (radius, radius))), window, axis=1).max(-1)


def fill_gaps_behind_objects(image: np.ndarray, distance: np.ndarray, max_px: int,
                             background_share: float) -> tuple[np.ndarray, np.ndarray]:
    """카메라가 움직여 앞 물체 뒤에서 드러난 틈을, 그 주변의 뒤쪽 배경 색을 늘려 메웁니다.

    틈은 앞 물체 모양을 그대로 닮아서, 비워 두면 생성 모델이 그 자리에 앞 물체를 한 번 더 그립니다.
    빈 픽셀마다 주변 max_px 안에서 가장 먼 깊이 × background_share보다 먼 이웃의 색만 한 픽셀씩 번지게 합니다.
    폭이 2 × max_px보다 넓은 빈 곳(원본 화면 밖)은 그대로 두고 생성 모델이 채웁니다.
    distance: (H, W) 깊이, 0은 빈 곳.
    """
    # ponytail: 배경 색을 늘리는 단순한 방식이라 메운 곳이 줄무늬로 보이지만, 생성 모델이 화면 전체를 다시 그리므로 그대로 둔다.
    if max_px <= 0:
        return image, distance
    far = local_max(distance, max_px)
    background = far * background_share
    # 양쪽이 막힌 좁은 틈만 메웁니다: 채워진 곳을 max_px만큼 넓혔다가 다시 줄였을 때 메워지는 곳 (닫힘 연산)
    filled = (distance > 0).astype(np.float32)
    narrow_gap = (1 - local_max(1 - local_max(filled, max_px), max_px)) > 0
    image, distance = image.copy(), distance.copy()
    for _ in range(max_px):
        empty = distance == 0
        if not empty.any():
            break
        best_depth = np.zeros_like(distance)
        best_color = np.zeros_like(image)
        padded_d = np.pad(distance, 1)
        padded_c = np.pad(image, ((1, 1), (1, 1), (0, 0)))
        h, w = distance.shape
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                d = padded_d[dy:dy + h, dx:dx + w]
                better = d > best_depth
                best_depth = np.where(better, d, best_depth)
                best_color = np.where(better[..., None], padded_c[dy:dy + h, dx:dx + w], best_color)
        grow = empty & narrow_gap & (best_depth > 0) & (best_depth >= background)
        if not grow.any():
            break
        distance[grow] = best_depth[grow]
        image[grow] = best_color[grow]
    return image, distance
