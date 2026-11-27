"""상품을 중심으로 옮긴 카메라 시점을 촬영의 참고 이미지로 만듭니다."""

from importlib.util import find_spec

from PIL import Image

from ..camera_angle_presets.settings import code_folder, load_angle_presets, load_settings, model_file


def available_viewpoints():
    if not code_folder("moge").is_dir() or find_spec("cv2") is None:
        return {}
    if find_spec("utils3d_moge") is None and find_spec("utils3d") is None:
        return {}
    try:
        model_file(load_settings()["depth"]["model"])
    except FileNotFoundError:
        return {}
    return {name: preset.label for name, preset in load_angle_presets().items() if preset.method == "orbit"}


def camera_view(photo, subject, preset):
    from ..camera_angle_presets.change_camera_angle import change_camera_angle, load_models, unload_models

    chosen = load_angle_presets().get(preset)
    if chosen is None or chosen.method != "orbit":
        raise ValueError("상품을 중심으로 옮길 수 있는 촬영 시점을 선택하세요")
    models = load_models(generation=False)
    try:
        views, _ = change_camera_angle(photo, [preset], models, subject_mask=subject)
        return views[0].image
    finally:
        unload_models(models)


def place_camera_view(view, source_box, source_size, picture_size, subject_box):
    left, top, right, bottom = source_box
    left, right = left * view.width / source_size[0], right * view.width / source_size[0]
    top, bottom = top * view.height / source_size[1], bottom * view.height / source_size[1]
    x, y, w, h = subject_box
    scale = min(w / (right - left), h / (bottom - top))
    rendered = view.resize((max(1, round(view.width * scale)), max(1, round(view.height * scale))),
                           Image.Resampling.LANCZOS)
    origin = (round(x + w / 2 - (left + right) / 2 * scale),
              round(y + h / 2 - (top + bottom) / 2 * scale))
    guide = Image.new("RGB", picture_size, load_settings()["orbit"]["hole_color"])
    guide.paste(rendered, origin)
    return guide
