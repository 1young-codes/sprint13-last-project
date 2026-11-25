"""고른 모델(PixRestore)로 사진을 키웁니다. 모델은 load_models()로 한 번 불러 두고, upscale(photo, models, scale)을
사진마다 부릅니다. 다 쓰면 unload_models(models)로 그래픽 메모리를 비웁니다.

추론 코드와 torch 버전이 모델 가상환경에 맞춰져 있어서, 그 환경의 파이썬으로 실행합니다.

    /data/venvs/upscales/pixrestore/bin/python -m models.generative_ai.upscales.upscale_photo 사진.jpg [사진2.jpg ...] --scale 2

모델을 만드는 순서는 PixRestore 저장소의 inference.py와 같고, 실행 값은 setup.toml [run] command를 그대로 씁니다.
조각 나누기와 이어 붙이기는 run_model.py와 같습니다.
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from .run_model import blend_tiles, tile_spots
from .settings import PROJECT_ROOT, code_folder, load_spec, project_path

MODEL = "pixrestore"


def load_models() -> dict:
    """한 번만 부릅니다. 복원 모델과 조건 특징을 뽑는 DINOv2를 그래픽 카드에 올립니다."""
    from accelerate import Accelerator

    folder = code_folder(MODEL)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import inference  # PixRestore 저장소의 추론 코드

    spec = load_spec(MODEL)
    saved_argv = sys.argv
    sys.argv = ["inference.py", *(part.format(input_dir="", output_dir="", scale=1) for part in spec["run"]["command"][1:])]
    try:
        args = inference.parse_args()  # 명령줄 실행과 같은 값을 쓰려고 그 해석기를 그대로 씀
    finally:
        sys.argv = saved_argv

    checkpoint = (folder / args.checkpoint).resolve()
    config_data = inference.read_config(inference.find_config_path(checkpoint, args.config))
    if args.cfg_scale is not None:
        config_data["cfg_scale"] = args.cfg_scale
    config_data["dinov2_repository"] = str(folder / args.dinov2_repository)
    config_data["dinov2_checkpoint"] = str(folder / args.dinov2_checkpoint)
    weight_path = inference.find_weights(checkpoint)
    state = inference.read_state_dict(weight_path)
    config = SimpleNamespace(**inference.apply_checkpoint_architecture_flags(config_data, state))

    accelerator = Accelerator(mixed_precision=str(config.mixed_precision or "no"))
    model = inference.build_model(config)
    inference.load_weights(model, weight_path, state=state)
    model.to(accelerator.device).eval().requires_grad_(False)
    encoder = None
    if config.use_venc:
        encoder = inference.load_dinov2(config.encoder_type, accelerator.device,
                                        repository=config.dinov2_repository, checkpoint=config.dinov2_checkpoint)
    models = {"inference": inference, "args": args, "config": config, "accelerator": accelerator, "model": model,
              "flow": inference.build_flow(config, accelerator), "encoder": encoder, "tiles": spec["run"]["tiles"]}
    # 추론 코드 일부(torch.compile)가 첫 실행 때 약 1분 동안 번역됩니다. 첫 요청이 느려지지 않게 불러올 때 미리 치러 둡니다.
    # 검은 조각으로는 번역이 일어나지 않아서 무늬 있는 조각을 씁니다.
    size = models["tiles"]["size"]
    restore_tile(models, Image.effect_noise((size, size), 64).convert("RGB"), args.seed)
    return models


def restore_tile(models: dict, tile: Image.Image, seed: int) -> Image.Image:
    import torch
    from torchvision.transforms.functional import pil_to_tensor, to_pil_image

    inference, args, config = models["inference"], models["args"], models["config"]
    accelerator, encoder = models["accelerator"], models["encoder"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lq = pil_to_tensor(tile).unsqueeze(0).to(accelerator.device).float().div_(127.5).sub_(1)
    with torch.no_grad(), accelerator.autocast():
        features = (inference.extract_layers(encoder, lq, config.encoder_layers, config.encoder_input_size)
                    if encoder is not None else None)
        restored = models["flow"].sample_multistep_fm(models["model"], lq, venc_fea=features,
                                                      n_steps=args.infer_steps, schedule=args.schedule)
    return to_pil_image(restored[0].float().cpu().add(1).mul(0.5).clamp(0, 1))


def upscale(photo: Image.Image, models: dict, scale: int) -> Image.Image:
    """photo를 바이큐빅으로 scale배 키운 뒤, 학습 해상도 조각으로 나눠 복원하고 겹쳐 이어 붙입니다."""
    size = models["tiles"]["size"]
    big = photo.convert("RGB").resize((photo.width * scale, photo.height * scale), Image.BICUBIC)
    spots = tile_spots(*big.size, models["tiles"])
    seed = models["args"].seed
    pieces = [restore_tile(models, big.crop((x, y, x + size, y + size)), seed + i) for i, (x, y) in enumerate(spots)]
    return Image.fromarray(blend_tiles(pieces, spots, *big.size, size))


def unload_models(models: dict) -> None:
    import gc

    import torch

    models.clear()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(prog="python -m models.generative_ai.upscales.upscale_photo",
                                     description="사진을 PixRestore로 키웁니다")
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--scale", type=int, default=2, choices=load_spec(MODEL)["run"]["scales"])
    parser.add_argument("--output", type=Path, help="결과 폴더 (기본: settings.toml output_dir/pixrestore)")
    args = parser.parse_args()
    output = args.output or project_path("output_dir") / MODEL
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    models = load_models()
    print(f"모델을 불러왔습니다: {time.perf_counter() - started:.1f}초", flush=True)
    for path in args.photos:
        started = time.perf_counter()
        with Image.open(path) as photo:
            upscale(photo, models, args.scale).save(output / f"{path.stem}.png")
        shown = output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output
        print(f"[완료] {path.name} → {shown / (path.stem + '.png')} ({time.perf_counter() - started:.1f}초)", flush=True)
    unload_models(models)


if __name__ == "__main__":
    main()
