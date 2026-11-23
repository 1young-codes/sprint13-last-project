"""카메라 앵글 계산 검증. 그래픽 카드와 모델 없이 돌아갑니다.

    python -m pytest tests/test_camera_angle.py
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from models.camera_angle.camera_math import CameraView, rotation_matrix_from_angles
from models.camera_angle.change_camera_angle import change_camera_angle, output_camera
from models.camera_angle.keep_original_detail import paste_original_where_visible
from models.camera_angle.move_camera_in_3d import orbit_camera_pose, render_from_moved_camera
from models.camera_angle.place_photo_in_360_view import place_photo_in_360_view
from models.camera_angle.rotate_camera import render_rotated_view
from models.camera_angle.settings import load_angle_presets

FOV = 60.0


def make_test_photo(width=240, height=160) -> Image.Image:
    """색 그라데이션 위에 가운데 빨간 점이 있는 시험용 사진."""
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    data = np.stack([x, y, 1 - x * y], axis=-1)
    image = Image.fromarray((data * 255).astype(np.uint8))
    ImageDraw.Draw(image).ellipse((width / 2 - 6, height / 2 - 6, width / 2 + 6, height / 2 + 6), fill=(255, 0, 0))
    return image


def red_dot_center(image: Image.Image) -> tuple[float, float]:
    data = np.asarray(image.convert("RGB")).astype(int)
    red = (data[..., 0] > 200) & (data[..., 1] < 60) & (data[..., 2] < 60)
    rows, cols = np.nonzero(red)
    return cols.mean(), rows.mean()


def view_of(photo, **angles) -> Image.Image:
    view, _ = place_photo_in_360_view(photo, FOV, 2048, "#00ff00")
    return render_rotated_view(view, CameraView(photo.width, photo.height, FOV, **angles))


def test_all_presets_load_and_make_valid_cameras():
    photo = make_test_photo()
    for preset in load_angle_presets().values():
        if preset.method == "rotate":
            output_camera(preset, photo, FOV)  # 화각과 렌즈 조합이 불가능하면 여기서 오류


def test_rotation_matrix_is_orthonormal():
    matrix = rotation_matrix_from_angles(30, -20, 15)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
    assert np.allclose(rotation_matrix_from_angles(0, 0, 0), np.eye(3))


def test_same_camera_returns_the_original():
    photo = make_test_photo()
    again = np.asarray(view_of(photo), dtype=float)
    difference = np.abs(again - np.asarray(photo, dtype=float))[4:-4, 4:-4].mean()
    assert difference < 4, difference


def test_original_paste_restores_full_detail():
    photo = make_test_photo()
    camera = CameraView(photo.width, photo.height, FOV)
    blurred = Image.new("RGB", photo.size, (0, 0, 0))
    pasted = paste_original_where_visible(blurred, photo, FOV, camera, feather_px=0)
    assert np.abs(np.asarray(pasted, dtype=float) - np.asarray(photo, dtype=float)).max() <= 2


def test_yaw_right_moves_subject_left_and_pitch_up_moves_it_down():
    photo = make_test_photo()
    cx, cy = red_dot_center(photo)
    x, _ = red_dot_center(view_of(photo, yaw_deg=10))
    _, y = red_dot_center(view_of(photo, pitch_deg=10))
    assert x < cx - 20 and y > cy + 20


def test_positive_roll_turns_scene_clockwise():
    """roll +20 결과는 PIL의 rotate(-20, 시계 방향)와 같아야 합니다."""
    photo = make_test_photo()
    rolled = np.asarray(view_of(photo, roll_deg=20), dtype=float)
    center = (slice(50, 110), slice(80, 160))
    same = np.abs(rolled - np.asarray(photo.rotate(-20, Image.Resampling.BILINEAR), dtype=float))[center].mean()
    opposite = np.abs(rolled - np.asarray(photo.rotate(20, Image.Resampling.BILINEAR), dtype=float))[center].mean()
    assert same < 6 < opposite, (same, opposite)


def flat_wall(photo, distance=2.0):
    """원본 카메라 앞 distance 거리에 사진이 붙은 벽이 있다고 보는 점 구름."""
    w, h = photo.size
    tangent = math.tan(math.radians(FOV) / 2)
    x, y = np.meshgrid((2 * (np.arange(w) + .5) / w - 1) * tangent, (1 - 2 * (np.arange(h) + .5) / h) * tangent * h / w)
    points = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32) * distance
    return points, np.ones((h, w), dtype=bool)


def test_orbit_without_angle_returns_the_original():
    photo = make_test_photo()
    points, valid = flat_wall(photo)
    moved, hole = render_from_moved_camera(photo, points, valid, FOV, 0, 0, 0.33, "#949494")
    assert np.asarray(hole).max() == 0
    assert np.abs(np.asarray(moved, dtype=float) - np.asarray(photo, dtype=float)).mean() < 1


def test_orbit_directions_match_preset_comments():
    center = np.array([0, 0, 2.0])
    _, left = orbit_camera_pose(center, 30, 0)
    _, below = orbit_camera_pose(center, 0, 25)
    assert left[0] < 0 and below[1] < 0  # orbit_yaw +: 카메라가 왼쪽으로, orbit_pitch +: 카메라가 아래로


def test_orbit_keeps_subject_center_in_place():
    photo = make_test_photo()
    points, valid = flat_wall(photo)
    moved, hole = render_from_moved_camera(photo, points, valid, FOV, 25, 0, 0.33, "#949494")
    x, y = red_dot_center(moved)
    assert abs(x - photo.width / 2) < 3 and abs(y - photo.height / 2) < 3
    assert np.asarray(hole).mean() > 0  # 비스듬히 보면 벽 바깥 빈 곳이 생깁니다


def test_rotate_presets_run_end_to_end_without_models():
    photo = make_test_photo()
    names = [name for name, preset in load_angle_presets().items() if preset.method == "rotate"]
    results, info = change_camera_angle(photo, names, photo_fov_deg=FOV, skip_generation=True)
    assert [r.preset.name for r in results] == names and info["photo_fov_deg"] == FOV
    original = next(r.image for r in results if r.preset.name == "original")
    assert original.size[0] == 1024 and abs(original.size[0] / original.size[1] - 1.5) < .01
