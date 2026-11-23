"""사진 한 장과 프리셋 이름들을 받아, 앵글을 바꾼 결과 이미지들을 돌려줍니다.

회전(rotate) 프리셋: 원본을 원하는 방향과 렌즈로 다시 찍기 → 원본 밖 빈 곳만 채우기
3D 이동(orbit) 프리셋: 깊이 추정 → 피사체 중심으로 카메라 돌려 그리기 → 원본을 참고해 구멍 채우기
"""

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image, ImageFilter

from .camera_math import CameraView
from .move_camera_in_3d import remove_depth_edge_points, render_from_moved_camera
from .rotate_camera import render_rotated_view
from .settings import AnglePreset, load_angle_presets, load_settings


@dataclass
class AngleResult:
    preset: AnglePreset
    image: Image.Image
    steps: dict[str, Image.Image] = field(default_factory=dict)  # 중간 단계 이미지 (확인용)


def output_size(photo: Image.Image, aspect: float | None, long_side: int) -> tuple[int, int]:
    aspect = aspect or photo.width / photo.height
    return (long_side, round(long_side / aspect)) if aspect >= 1 else (round(long_side * aspect), long_side)


def output_camera(preset: AnglePreset, photo: Image.Image, photo_fov_deg: float) -> CameraView:
    settings = load_settings()["photo"]
    width, height = output_size(photo, preset.aspect, settings["output_long_side"])
    vertical_fov_deg = math.degrees(2 * math.atan(math.tan(math.radians(photo_fov_deg) / 2) * photo.height / photo.width))
    fov = preset.output_fov_deg(photo_fov_deg, settings["max_fov_deg"], settings["max_curved_fov_deg"])
    return CameraView(width=width, height=height, fov_deg=fov,
                      yaw_deg=preset.yaw_deg + preset.yaw_ratio * photo_fov_deg,
                      pitch_deg=preset.pitch_deg + preset.pitch_ratio * vertical_fov_deg,
                      roll_deg=preset.roll_deg, lens_curve=preset.lens_curve)


def seen_share(empty_mask: Image.Image) -> float:
    """결과 화면에서 원본이 보이는 비율."""
    return 1 - float(np.asarray(empty_mask, dtype=np.float32).mean()) / 255


def widen_photo(photo: Image.Image, photo_fov_deg: float, target: CameraView, scene_description: str, seed: int,
                steps: int, model) -> tuple[Image.Image, float, list[Image.Image]]:
    """(넓힌 사진, 넓힌 화각, 단계별 그림). 원본이 target 화면의 일부만 차지하면 생성 모델이 한 번에 채우지 못하므로
    (빈 곳이 너무 크면 투명 배경으로 그려 버림), 정면 화각을 몇 배씩 넓혀 가며 여러 번 나눠 채웁니다."""
    from .fill_empty_area import fill_empty_area

    settings = load_settings()
    rotate, long_side = settings["rotate"], settings["photo"]["output_long_side"]
    stages = []
    while True:
        _, empty = render_rotated_view(photo, photo_fov_deg, target, rotate["empty_color"])
        # 목표 화각까지 같은 비율로 나눠 넓힙니다. 한 단계를 꽉 채워 넓히면 마지막에 얇은 띠만 남는데,
        # 생성 모델은 얇은 띠를 지저분한 무늬로 채웁니다.
        ratio = min(target.fov_deg, settings["photo"]["max_fov_deg"]) / photo_fov_deg
        scale = rotate["widen_step_scale"]
        if ratio > 1.001:
            scale = ratio ** (1 / math.ceil(math.log(ratio) / math.log(scale)))
        wider = min(photo_fov_deg * scale, settings["photo"]["max_fov_deg"])
        if seen_share(empty) >= rotate["min_original_share"] or wider <= photo_fov_deg + 1e-3:
            return photo, photo_fov_deg, stages
        width, height = output_size(photo, None, long_side)
        rendered, empty = render_rotated_view(photo, photo_fov_deg, CameraView(width, height, wider), rotate["empty_color"])
        photo = fill_empty_area(model, rendered, empty, scene_description, seed, steps)
        photo_fov_deg = wider
        stages.append(photo)


def render_and_fill(photo: Image.Image, photo_fov_deg: float, camera: CameraView, scene_description: str, seed: int,
                    steps: int, model_getter) -> tuple[Image.Image, dict[str, Image.Image]]:
    """(결과, 중간 단계). 원본을 camera로 다시 찍고 빈 곳을 채웁니다.

    1. 원본이 화면의 작은 일부뿐이면 정면 화각을 나눠서 넓혀 둡니다 (widen_photo).
    2. 어안처럼 휘는 렌즈는 생성 모델이 휜 그림을 이어 그리지 못하므로, 같은 방향의 일반 렌즈로 찍어 채운 뒤 휘게 만듭니다.
    3. 마지막에 원본이 보이는 곳은 원본에서 직접 다시 찍어 덮어, 넓히는 동안 줄어든 해상도를 되살립니다.
    """
    settings = load_settings()
    color = settings["rotate"]["empty_color"]
    flat = camera if camera.lens_curve == 1 else replace(
        camera, fov_deg=min(camera.fov_deg, settings["photo"]["max_fov_deg"]), lens_curve=1.0)
    source, source_fov, stages = photo, photo_fov_deg, []
    if model_getter is not None:
        source, source_fov, stages = widen_photo(photo, photo_fov_deg, flat, scene_description, seed, steps,
                                                 model_getter())
    rendered, empty_mask = render_rotated_view(source, source_fov, flat, color)
    steps_shown = {f"widened_{i + 1}": stage for i, stage in enumerate(stages)}
    steps_shown.update(before_fill=rendered, empty_mask=empty_mask)
    if model_getter is None:
        return rendered, steps_shown
    from .fill_empty_area import fill_empty_area
    image = fill_empty_area(model_getter(), rendered, empty_mask, scene_description, seed, steps)
    if flat is not camera:
        steps_shown["before_lens_curve"] = image
        image, _ = render_rotated_view(image, flat.fov_deg, replace(camera, yaw_deg=0, pitch_deg=0, roll_deg=0), "#000000")
    direct, unseen = render_rotated_view(photo, photo_fov_deg, camera, "#000000")
    feather = settings["rotate"]["feather_px"]
    if feather > 0:
        unseen = unseen.filter(ImageFilter.MaxFilter(2 * feather + 1)).filter(ImageFilter.GaussianBlur(feather / 2))
    return Image.composite(image, direct, unseen), steps_shown


def select_presets(names: list[str]) -> list[AnglePreset]:
    presets = load_angle_presets()
    if unknown := [n for n in names if n not in presets]:
        raise ValueError(f"없는 프리셋입니다: {unknown}. 사용할 수 있는 프리셋: {list(presets)}")
    return [presets[n] for n in names]


def change_camera_angle(photo: Image.Image, preset_names: list[str], scene_description: str = "",
                        seed: int | None = None, steps: int | None = None, photo_fov_deg: float | None = None,
                        skip_generation: bool = False, model=None) -> tuple[list[AngleResult], dict]:
    """(결과 목록, 실행 정보). skip_generation이면 생성 모델 없이 빈 곳을 표시색으로 남겨 계산만 확인합니다.
    model을 주면 그 생성 모델을 씁니다 (여러 사진을 이어서 처리할 때 한 번만 불러오도록)."""
    settings = load_settings()
    presets = select_presets(preset_names)
    photo = photo.convert("RGB")
    seed = settings["generation"]["seed"] if seed is None else seed
    steps = steps or settings["generation"]["steps"]
    photo_fov_deg = photo_fov_deg or settings["photo"]["fov_deg"] or None

    depth = None
    if photo_fov_deg is None or any(p.method == "orbit" for p in presets):
        from .estimate_depth import estimate_depth_and_fov
        depth = estimate_depth_and_fov(photo)
        photo_fov_deg = photo_fov_deg or depth.fov_deg

    def get_model():
        nonlocal model
        if model is None:
            from .qwen_image_model import QwenImageModel
            model = QwenImageModel()
        return model

    model_getter = None if skip_generation else get_model
    results, seconds = [], {}
    for preset in (p for p in presets if p.method == "rotate"):
        started = time.perf_counter()
        image, steps_shown = render_and_fill(photo, photo_fov_deg, output_camera(preset, photo, photo_fov_deg),
                                             scene_description, seed, steps, model_getter)
        results.append(AngleResult(preset, image, steps_shown))
        seconds[preset.name] = round(time.perf_counter() - started, 1)
    orbit = settings["orbit"]
    orbit_presets = [p for p in presets if p.method == "orbit"]
    if orbit_presets:
        solid_points = remove_depth_edge_points(depth.points, depth.valid, orbit["depth_edge_tolerance"])
    for preset in orbit_presets:
        started = time.perf_counter()
        moved, hole_mask = render_from_moved_camera(photo, depth.points, solid_points, depth.fov_deg,
                                                    preset.orbit_yaw_deg, preset.orbit_pitch_deg,
                                                    orbit["subject_crop"], orbit["hole_color"], orbit["point_size_px"],
                                                    orbit["gap_fill_ratio"], orbit["background_depth_share"])
        size = output_size(photo, None, settings["photo"]["output_long_side"])
        moved = moved.resize(size, Image.Resampling.LANCZOS)
        image = moved
        if model_getter is not None:
            from .complete_new_viewpoint import complete_view_with_reference
            image = complete_view_with_reference(model_getter(), photo, moved, seed, steps)
        results.append(AngleResult(preset, image, {"moved_view": moved, "hole_mask": hole_mask}))
        seconds[preset.name] = round(time.perf_counter() - started, 1)

    info = {"presets": preset_names, "photo_fov_deg": photo_fov_deg, "seed": seed, "steps": steps,
            "scene_description": scene_description, "skip_generation": skip_generation,
            "depth_fov_deg": depth.fov_deg if depth else None,
            "seconds_per_preset": seconds}  # 생성 모델을 처음 불러오는 시간은 첫 프리셋에 들어갑니다
    return results, info
