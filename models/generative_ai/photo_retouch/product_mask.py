"""BiRefNet으로 상품(앞쪽 피사체) 영역을 0~1 마스크로 만듭니다."""

import numpy as np

from .download_models import model_folder
from .settings import load_settings

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def product_mask(image: np.ndarray) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageSegmentation

    size = load_settings()["product_mask"]["input_size"]
    model = AutoModelForImageSegmentation.from_pretrained(model_folder("product_mask"), trust_remote_code=True)
    model.to("cuda").eval()
    small = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), np.float32) / 255
    x = torch.from_numpy((small - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1)[None].to("cuda")
    with torch.no_grad():
        prediction = model(x)[-1].sigmoid()[0, 0].float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    height, width = image.shape[:2]
    return np.asarray(Image.fromarray(prediction).resize((width, height), Image.BILINEAR), np.float32).clip(0, 1)
