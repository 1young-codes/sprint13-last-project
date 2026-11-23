"""360도 배경에서 사진이 덮지 못한 빈 곳(초록색)을 Qwen + 360도 확장 LoRA로 채웁니다."""

from PIL import Image

from .settings import load_settings


def fill_empty_area_of_360_view(model, view_360: Image.Image, empty_mask: Image.Image, scene_description: str,
                                seed: int, steps: int) -> Image.Image:
    """빈 곳만 생성 결과로 바꾸고, 원본 사진이 있던 곳은 그대로 둡니다."""
    settings = load_settings()["view_360"]
    model.use_lora("fill_360_lora", settings["fill_lora_strength"])
    prompt = settings["fill_prompt"].format(scene=scene_description).strip()
    filled = model.edit(view_360, prompt, seed, steps).resize(view_360.size, Image.Resampling.LANCZOS)
    return Image.composite(filled, view_360.convert("RGB"), empty_mask.convert("L"))
