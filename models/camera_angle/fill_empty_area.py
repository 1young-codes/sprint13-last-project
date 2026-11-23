"""다시 찍은 사진에서 원본이 찍지 않은 빈 곳(표시색)을 생성 모델로 채웁니다."""

from PIL import Image, ImageFilter

from .settings import load_settings


def fill_empty_area(model, image: Image.Image, empty_mask: Image.Image, scene_description: str,
                    seed: int, steps: int) -> Image.Image:
    """빈 곳만 생성 결과로 바꾸고, 원본 픽셀은 그대로 둡니다. 빈 곳이 없으면 생성하지 않습니다."""
    settings = load_settings()["rotate"]
    if not empty_mask.getbbox():
        return image
    model.use_lora(None, 0)
    prompt = settings["fill_prompt"].format(scene=scene_description).strip()
    filled = model.edit(image, prompt, seed, steps).resize(image.size, Image.Resampling.LANCZOS)
    # 빈 곳을 조금 넓힌 뒤 흐리게 해서, 원본과 생성 영역의 경계선이 보이지 않게 섞습니다
    feather = settings["feather_px"]
    mask = empty_mask.convert("L")
    if feather > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * feather + 1)).filter(ImageFilter.GaussianBlur(feather / 2))
    return Image.composite(filled, image.convert("RGB"), mask)
