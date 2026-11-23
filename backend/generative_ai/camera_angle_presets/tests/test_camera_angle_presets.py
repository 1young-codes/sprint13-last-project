"""카메라 앵글 계산 검증. 그래픽 카드와 모델 없이 돌아갑니다.

    python -m pytest tests/test_camera_angle.py
"""

import math
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from models.camera_angle.camera_math import CameraView, rotation_matrix_from_angles
from models.camera_angle.change_camera_angle import change_camera_angle, output_camera
from models.camera_angle.move_camera_in_3d import orbit_camera_pose, render_from_moved_camera
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
    return render_rotated_view(photo, FOV, CameraView(photo.width, photo.height, FOV, **angles), "#00ff00")[0]


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


def test_empty_mask_marks_only_what_the_original_did_not_see():
    photo = make_test_photo()
    _, same = render_rotated_view(photo, FOV, CameraView(photo.width, photo.height, FOV), "#00ff00")
    _, wider = render_rotated_view(photo, FOV, CameraView(photo.width, photo.height, 90), "#00ff00")
    wider = np.asarray(wider)
    assert np.asarray(same)[2:-2, 2:-2].max() == 0
    assert wider[:, :10].min() == 255 and wider[60:100, 100:140].max() == 0  # 가장자리만 비고 가운데는 원본


def test_fisheye_outside_the_lens_circle_is_black_not_empty():
    photo = make_test_photo()
    image, empty = render_rotated_view(photo, FOV, CameraView(200, 200, 300, lens_curve=0), "#00ff00")  # 모서리는 렌즈 원 밖
    assert np.asarray(empty)[0, 0] == 0 and np.asarray(image)[0, 0].max() == 0


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


def test_depth_edge_points_are_removed_but_flat_surfaces_stay():
    from models.camera_angle.move_camera_in_3d import remove_depth_edge_points
    photo = make_test_photo()
    points, valid = flat_wall(photo)
    assert remove_depth_edge_points(points, valid, 0.04).all()
    points[:, photo.width // 2:] *= 2  # 오른쪽 절반을 두 배 멀리: 가운데에 깊이 경계
    kept = remove_depth_edge_points(points, valid, 0.04)
    assert not kept[:, photo.width // 2 - 1:photo.width // 2 + 1].any() and kept[:, :photo.width // 2 - 2].all()


def test_gap_behind_an_object_is_filled_with_the_background_not_the_object():
    from models.camera_angle.move_camera_in_3d import fill_gaps_behind_objects
    image = np.zeros((20, 30, 3), dtype=np.float32)
    distance = np.zeros((20, 30), dtype=np.float32)
    image[:, :12], distance[:, :12] = (1, 0, 0), 1.0   # 왼쪽: 가까운 빨간 물체
    image[:, 15:], distance[:, 15:] = (0, 0, 1), 3.0   # 오른쪽: 먼 파란 배경, 그 사이 3픽셀이 드러난 틈
    filled, depth = fill_gaps_behind_objects(image, distance, 4, 0.9)
    assert (depth[:, 12:15] == 3.0).all() and np.allclose(filled[:, 12:15], (0, 0, 1))
    wide = distance.copy()
    wide[:, 12:] = 0  # 오른쪽 전체가 빈 넓은 곳은 메우지 않습니다
    assert (fill_gaps_behind_objects(image, wide, 4, 0.9)[1][:, 12:] == 0).all()


def test_generated_colors_are_matched_to_the_original_near_the_border():
    from models.camera_angle.fill_empty_area import match_colors_to_original
    original, generated = np.full((40, 60, 3), .8), np.full((40, 60, 3), .6)  # 생성 모델이 조금 어둡게 그림
    known = np.zeros((40, 60))
    known[:, :30] = 1  # 왼쪽 절반이 원본
    matched = match_colors_to_original(generated, original, known, 4)
    assert abs(matched[20, 32, 0] - .8) < .02  # 경계 바로 바깥은 원본 밝기로 이어짐


def test_pan_and_tilt_follow_the_photo_field_of_view():
    """좌우·위아래 돌리기는 원본 화각에 비례해, 화각이 좁은 사진에서도 피사체가 화면에 남습니다."""
    presets = load_angle_presets()
    for fov in (40.0, 80.0):
        photo = make_test_photo()
        camera = output_camera(presets["pan_right"], photo, fov)
        assert abs(camera.yaw_deg - presets["pan_right"].yaw_ratio * fov) < 1e-6 and camera.yaw_deg < fov / 2
        x, _ = red_dot_center(render_rotated_view(photo, fov, replace(camera, width=240, height=160), "#00ff00")[0])
        assert 0 < x < 120  # 피사체가 왼쪽으로 갔지만 화면 안에 있음


class EchoModel:
    """생성 대신 받은 그림을 그대로 돌려주는 가짜 모델 (그래픽 카드 없이 흐름만 확인)."""

    def use_lora(self, model_key, strength):
        pass

    def edit(self, image, prompt, seed, steps):
        return image.convert("RGB")


def test_far_wider_views_are_filled_in_stages():
    """원본이 아주 작게 남는 화각은 여러 번 나눠 채우고, 매 단계 원본이 정해진 비율 이상 남습니다."""
    from models.camera_angle.change_camera_angle import widen_photo
    from models.camera_angle.settings import load_settings
    min_share = load_settings()["rotate"]["min_original_share"]
    seen_shares = []

    class RecordingModel(EchoModel):
        def edit(self, image, prompt, seed, steps):
            a = np.asarray(image.convert("RGB")).copy()
            green = (a[..., 1] > 200) & (a[..., 0] < 60) & (a[..., 2] < 60)
            seen_shares.append(1 - green.mean())
            a[green] = 128  # 빈 곳을 회색으로 채운 척
            return Image.fromarray(a)

    photo = make_test_photo()
    _, fov, stages = widen_photo(photo, FOV, CameraView(240, 160, 150), "", 0, 1, RecordingModel())
    assert len(stages) >= 2 and FOV < fov <= 150
    assert min(seen_shares) >= min_share - 0.03, seen_shares
    assert widen_photo(photo, FOV, CameraView(240, 160, 66), "", 0, 1, EchoModel())[2] == []  # 조금 넓히면 한 번에
