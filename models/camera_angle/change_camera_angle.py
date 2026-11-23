"""사진 한 장과 프리셋 이름들을 받아, 앵글을 바꾼 결과 이미지들을 돌려줍니다.

회전(rotate) 프리셋: 사진을 360도 배경에 붙임 → 빈 곳 채우기 → 원하는 방향으로 다시 찍기 → 원본 픽셀 덮기
3D 이동(orbit) 프리셋: 깊이 추정 → 피사체 중심으로 카메라 돌려 그리기 → 원본을 참고해 구멍 채우기

360도 배경은 사진마다 한 번만 만들고 캐시에 저장하므로, 회전 프리셋을 여러 개 골라도 생성은 한 번입니다.
"""

import hashlib
import json
from dataclasses import dataclass, field

from PIL import Image

from .camera_math import CameraView
from .keep_original_detail import paste_original_where_visible
from .move_camera_in_3d import remove_depth_edge_points, render_from_moved_camera
from .place_photo_in_360_view import place_photo_in_360_view
from .rotate_camera import render_rotated_view
from .settings import AnglePreset, load_angle_presets, load_settings, project_path


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
    return CameraView(width=width, height=height, fov_deg=preset.output_fov_deg(photo_fov_deg, settings["max_fov_deg"]),
                      yaw_deg=preset.yaw_deg, pitch_deg=preset.pitch_deg, roll_deg=preset.roll_deg,
                      lens_curve=preset.lens_curve)


def select_presets(names: list[str]) -> list[AnglePreset]:
    presets = load_angle_presets()
    if unknown := [n for n in names if n not in presets]:
        raise ValueError(f"없는 프리셋입니다: {unknown}. 사용할 수 있는 프리셋: {list(presets)}")
    return [presets[n] for n in names]


def build_360_view(photo: Image.Image, photo_fov_deg: float, scene_description: str, seed: int, steps: int,
                   get_model) -> tuple[Image.Image, Image.Image]:
    """(채워진 360도 배경, 채우기 전 빈 곳 마스크). get_model이 None이면 생성 없이 빈 곳을 표시색으로 둡니다."""
    settings = load_settings()["view_360"]
    placed, empty_mask = place_photo_in_360_view(photo, photo_fov_deg, settings["width"], settings["empty_color"])
    if get_model is None:
        return placed, empty_mask
    recipe = json.dumps([photo_fov_deg, scene_description, seed, steps, settings], sort_keys=True, ensure_ascii=False)
    key = hashlib.sha256(photo.convert("RGB").tobytes() + recipe.encode()).hexdigest()[:20]
    cached = project_path("cache_dir") / f"view_360_{key}.png"
    if cached.is_file():
        return Image.open(cached).convert("RGB"), empty_mask
    from .fill_empty_area import fill_empty_area_of_360_view
    from .repair_360_seam import repair_left_right_seam
    filled = fill_empty_area_of_360_view(get_model(), placed, empty_mask, scene_description, seed, steps)
    filled = repair_left_right_seam(get_model(), filled, seed, steps)
    cached.parent.mkdir(parents=True, exist_ok=True)
    filled.save(cached)
    return filled, empty_mask


def change_camera_angle(photo: Image.Image, preset_names: list[str], scene_description: str = "",
                        seed: int | None = None, steps: int | None = None, photo_fov_deg: float | None = None,
                        skip_generation: bool = False) -> tuple[list[AngleResult], dict]:
    """(결과 목록, 실행 정보). skip_generation이면 생성 모델 없이 빈 곳을 표시색으로 남겨 계산만 확인합니다."""
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

    model = None

    def get_model():
        nonlocal model
        if model is None:
            from .qwen_image_model import QwenImageModel
            model = QwenImageModel()
        return model

    model_getter = None if skip_generation else get_model
    results = []
    rotate_presets = [p for p in presets if p.method == "rotate"]
    if rotate_presets:
        view_360, empty_mask = build_360_view(photo, photo_fov_deg, scene_description, seed, steps, model_getter)
        for preset in rotate_presets:
            camera = output_camera(preset, photo, photo_fov_deg)
            rendered = render_rotated_view(view_360, camera)
            image = paste_original_where_visible(rendered, photo, photo_fov_deg, camera,
                                                 settings["view_360"]["keep_original_feather_px"])
            results.append(AngleResult(preset, image, {"view_360": view_360, "view_360_empty_mask": empty_mask,
                                                       "before_original_paste": rendered}))
    orbit = settings["orbit"]
    orbit_presets = [p for p in presets if p.method == "orbit"]
    if orbit_presets:
        solid_points = remove_depth_edge_points(depth.points, depth.valid, orbit["depth_edge_tolerance"])
    for preset in orbit_presets:
        moved, hole_mask = render_from_moved_camera(photo, depth.points, solid_points, depth.fov_deg,
                                                    preset.orbit_yaw_deg, preset.orbit_pitch_deg,
                                                    orbit["subject_crop"], orbit["hole_color"], orbit["point_size_px"])
        size = output_size(photo, None, settings["photo"]["output_long_side"])
        moved = moved.resize(size, Image.Resampling.LANCZOS)
        image = moved
        if model_getter is not None:
            from .complete_new_viewpoint import complete_view_with_reference
            image = complete_view_with_reference(model_getter(), photo, moved, seed, steps)
        results.append(AngleResult(preset, image, {"moved_view": moved, "hole_mask": hole_mask}))

    info = {"presets": preset_names, "photo_fov_deg": photo_fov_deg, "seed": seed, "steps": steps,
            "scene_description": scene_description, "skip_generation": skip_generation,
            "depth_fov_deg": depth.fov_deg if depth else None}
    return results, info
