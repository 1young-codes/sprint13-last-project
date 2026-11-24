"""카메라를 3D로 옮긴 뒤 드러난 구멍을, 원본 사진을 참고 이미지로 함께 넣어 Qwen + AnyAngle LoRA로 채웁니다."""

from PIL import Image

from .settings import load_settings, model_file


def complete_view_with_reference(model, original: Image.Image, moved_view: Image.Image, seed: int,
                                 steps: int) -> Image.Image:
    """원본 예제처럼 화면 전체를 다시 그립니다. 결과 크기는 moved_view와 같습니다."""
    settings = load_settings()["orbit"]
    model.use_lora(model_file("new_viewpoint_lora"), settings["lora_strength"])
    completed = model.edit_with_reference(original, moved_view, settings["prompt"], seed, steps)
    return completed.resize(moved_view.size, Image.Resampling.LANCZOS)
