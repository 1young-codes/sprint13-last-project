"""Generate comparison ads using the prompts recorded by the Qwen run."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps

WORKFLOWS = {
    "moss": "MOSS-Image-SFT BF16 + Moss-Image-Edit, Euler/normal 30, CFG 4, SLG 22",
    "flux": "FLUX.3 [klein] FP8, native reference conditioning, 16 steps, prompt as provided",
}


def generate(args, jobs):
    if args.model == "moss":
        from . import reference_moss

        return reference_moss.generate_batch(jobs, args.moss_snapshot, args.weights / "moss")
    from . import reference_flux

    return reference_flux.generate_batch(jobs, args.weights / "flux")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=WORKFLOWS)
    parser.add_argument("qwen_settings", type=Path)
    parser.add_argument("photos", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--moss-snapshot", type=Path)
    args = parser.parse_args()
    if args.model == "moss" and args.moss_snapshot is None:
        parser.error("--moss-snapshot is required for MOSS")

    base = json.loads(args.qwen_settings.read_text(encoding="utf-8"))
    records = base["ads"]
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [record for record in records if not (args.output / record["file"]).is_file()]
    jobs = []
    for record in pending:
        with Image.open(args.photos / record["reference"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        jobs.append((base["prompt_template"].format(**record), image))

    images = generate(args, jobs) if jobs else ()
    start = perf_counter()
    for record, image in zip(pending, images, strict=True):
        target = args.output / record["file"]
        temporary = target.with_suffix(".tmp")
        image.save(temporary, format="PNG")
        temporary.replace(target)
        print(f"{target.name} {image.width}x{image.height} at {perf_counter() - start:.1f}s", flush=True)
    settings = {
        "model": args.model, "workflow": WORKFLOWS[args.model], "seed": 1,
        "prompt_template": base["prompt_template"],
        "ads": [{key: ad[key] for key in ("file", "reference", "title", "price")} for ad in records],
    }
    (args.output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
