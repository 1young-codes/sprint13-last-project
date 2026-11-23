"""카메라 방향과 픽셀 좌표 계산 (numpy만 사용).

좌표계: x 오른쪽, y 위, z 앞. 360도 이미지는 가로가 경도(-180°~180°), 세로가 위도(90°~-90°)입니다.
계산 방식과 픽셀 중심 규약은 ComfyUI-Panorama-Stickers(MIT, 커밋 6db991a)를 따르며,
ai-generation 프로젝트의 backend/studio/panorama.py에서 옮겨 왔습니다. 고지문은 LICENSE-Panorama-Stickers.txt에 있습니다.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CameraView:
    """한 장의 사진을 찍는 카메라. 세로 화각은 가로 화각과 이미지 비율로 정해집니다."""
    width: int
    height: int
    fov_deg: float
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    lens_curve: float = 1.0  # 1 일반 렌즈, 0.5 입체 투영, 0 어안

    def __post_init__(self):
        if self.fov_deg * self.lens_curve > 179:
            raise ValueError(f"화각 {self.fov_deg}°는 lens_curve {self.lens_curve}에서 만들 수 없습니다 (화각 × lens_curve ≤ 179)")


def rotation_matrix_from_angles(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """열이 각각 카메라의 오른쪽, 위, 앞 방향인 3×3 회전 행렬."""
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cy * cr - sy * sp * sr, -cy * sr - sy * sp * cr, sy * cp],
                     [cp * sr, cp * cr, sp],
                     [-sy * cr - cy * sp * sr, sy * sr - cy * sp * cr, cy * cp]], dtype=np.float32)


def camera_rotation(camera: CameraView) -> np.ndarray:
    return rotation_matrix_from_angles(camera.yaw_deg, camera.pitch_deg, camera.roll_deg)


def pixel_directions_for_photo(camera: CameraView, start_row: int, stop_row: int) -> np.ndarray:
    """카메라 사진의 start_row~stop_row 행에서 각 픽셀이 바라보는 방향 (행, 열, 3). 화각 밖은 0 벡터입니다."""
    w, h = camera.width, camera.height
    if camera.lens_curve != 1:
        curve = camera.lens_curve
        half = math.radians(camera.fov_deg) / 2
        focal = w / (2 * (math.tan(curve * half) / curve if curve else half))
        x, y = np.meshgrid((np.arange(w) + .5 - w / 2) / focal, (h / 2 - np.arange(start_row, stop_row) - .5) / focal)
        radius = np.hypot(x, y)
        theta = np.arctan(curve * radius) / curve if curve else radius
        factor = np.divide(np.sin(theta), radius, out=np.ones_like(radius), where=radius > 0)
        rays = np.stack((x * factor, y * factor, np.cos(theta)), axis=-1).astype(np.float32)
        rays[theta > math.pi] = 0
        return rays @ camera_rotation(camera).T
    tangent = math.tan(math.radians(camera.fov_deg) / 2)
    vertical = tangent * h / w
    x = (2 * (np.arange(w, dtype=np.float32) + .5) / w - 1) * tangent
    y = (1 - 2 * (np.arange(start_row, stop_row, dtype=np.float32) + .5) / h) * vertical
    xx, yy = np.broadcast_arrays(x[None, :], y[:, None])
    rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays @ camera_rotation(camera).T


def pixel_directions_for_360_image(width: int, height: int, start_row: int, stop_row: int) -> np.ndarray:
    """360도 이미지의 start_row~stop_row 행에서 각 픽셀이 가리키는 방향 (행, 열, 3)."""
    longitude = ((np.arange(width, dtype=np.float32) + .5) / width - .5) * 2 * math.pi
    latitude = (.5 - (np.arange(start_row, stop_row, dtype=np.float32) + .5) / height) * math.pi
    lon, lat = np.broadcast_arrays(longitude[None, :], latitude[:, None])
    return np.stack((np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)), axis=-1)


def to_premultiplied_rgba(image: Image.Image) -> np.ndarray:
    """0~1 실수 RGBA. 색에 투명도를 미리 곱해 두어야 가장자리 색이 번지지 않습니다."""
    data = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
    data[..., :3] *= data[..., 3:4]
    return data


def sample_pixels_bilinear(data: np.ndarray, x: np.ndarray, y: np.ndarray, wrap_x: bool = False) -> np.ndarray:
    """실수 좌표 (x, y)의 색을 주변 네 픽셀로 보간합니다. wrap_x면 좌우 끝이 이어집니다(360도 이미지)."""
    h, w = data.shape[:2]
    x = np.mod(x.astype(np.float64), w) if wrap_x else np.clip(x, 0, w - 1)
    if wrap_x:
        x = np.where(x >= w, 0, x)  # 아주 작은 음수의 나머지가 w로 반올림되는 경우
    y = np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1 = (x0 + 1) % w if wrap_x else np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (data[y0, x0] * (1 - fx) + data[y0, x1] * fx) * (1 - fy) + (data[y1, x0] * (1 - fx) + data[y1, x1] * fx) * fy


def read_colors_from_360_image(image: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """방향마다 360도 이미지에서 색을 읽습니다."""
    lon = np.arctan2(directions[..., 0], directions[..., 2])
    lat = np.arcsin(np.clip(directions[..., 1], -1, 1))
    x = (lon / (2 * math.pi) + .5) * image.shape[1] - .5
    y = (.5 - lat / math.pi) * image.shape[0] - .5
    return sample_pixels_bilinear(image, x, y, wrap_x=True)


def read_colors_from_photo(photo: np.ndarray, directions: np.ndarray, camera: CameraView) -> np.ndarray:
    """방향마다 일반 렌즈 사진에서 색을 읽습니다. 사진이 찍지 않은 방향은 투명(0)입니다."""
    local = directions @ camera_rotation(camera)
    z = np.maximum(local[..., 2], 1e-8)
    tangent = math.tan(math.radians(camera.fov_deg) / 2)
    u = local[..., 0] / z / tangent
    v = -local[..., 1] / z / (tangent * camera.height / camera.width)
    visible = (local[..., 2] > 0) & (np.abs(u) <= 1) & (np.abs(v) <= 1)
    x = (u + 1) / 2 * photo.shape[1] - .5
    y = (v + 1) / 2 * photo.shape[0] - .5
    return sample_pixels_bilinear(photo, x, y) * visible[..., None]


def hex_to_rgb(value: str) -> np.ndarray:
    return np.array([int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=np.float32)


def to_uint8_image(data: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(data, 0, 1) * 255).astype(np.uint8))
