"""모델 하나로 폴더 안 사진을 업스케일하고, 걸린 시간과 최대 그래픽 메모리를 잽니다.

    python -m models.generative_ai.upscales.run_model vosr --input 사진폴더 --scale 4

모델은 자기 가상환경의 파이썬으로 따로 실행되고(setup.toml [run] command), 이 모듈은 바깥에서 지켜봅니다.
결과는 모델마다 이름 규칙이 달라서, 원본 이름 그대로 <출력 폴더>/<원본 이름>.png로 모아 둡니다.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image

from .settings import PROJECT_ROOT, code_folder, load_settings, load_spec, project_path, venv_python

RAW_FOLDER = "raw"  # 모델이 직접 쓴 결과


def list_images(folder: Path) -> list[Path]:
    suffixes = load_settings()["run"]["image_suffixes"]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in suffixes)


def build_command(name: str, spec: dict, input_dir: Path, output_dir: Path, scale: int, images: list[Path]) -> list[str]:
    command = spec["run"]["command"]
    text = " ".join(command)
    values = {"input_dir": input_dir.resolve(), "output_dir": output_dir.resolve(), "scale": scale}
    if "{input_file}" in text or "{input_stem}" in text:  # 한 번에 한 장만 받는 모델 (VARestorer)
        if len(images) != 1:
            raise ValueError(f"{name}: 한 번에 사진 한 장만 받는 모델입니다. 폴더에 한 장만 넣으세요")
        values |= {"input_file": images[0].resolve(), "input_stem": images[0].stem}
    if "{out_short}" in text:  # 배율 대신 출력 짧은 변을 받는 모델 (SeedVR2)
        sizes = {Image.open(p).size for p in images}
        if len(sizes) != 1:
            raise ValueError(f"{name}: 출력 크기를 하나로 받는 모델이라 크기가 같은 사진만 한 번에 넣을 수 있습니다")
        values["out_short"] = min(sizes.pop()) * scale
    return [str(venv_python(name)), *(part.format(**values) for part in command)]


def watch(command: list[str], cwd: Path, env: dict, log_path: Path) -> tuple[float, float, int]:
    """명령을 실행하며 그 프로세스와 자식 프로세스의 그래픽 메모리 합의 최대값(MB)을 잽니다."""
    import psutil
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    interval = load_settings()["run"]["vram_poll_seconds"]
    peak = 0.0
    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        parent = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                pids = {process.pid, *(child.pid for child in parent.children(recursive=True))}
            except psutil.NoSuchProcess:
                pids = {process.pid}
            used = sum(p.usedGpuMemory or 0 for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid in pids)
            peak = max(peak, used / 2**20)
            time.sleep(interval)
    seconds = time.perf_counter() - start
    pynvml.nvmlShutdown()
    return seconds, peak, process.returncode


def find_output(raw_dir: Path, stem: str) -> Path | None:
    """모델이 쓴 결과 중 원본 이름이 들어간 가장 큰 이미지."""
    suffixes = load_settings()["run"]["image_suffixes"]
    found = [p for p in raw_dir.rglob("*") if p.suffix.lower() in suffixes and stem in p.stem]
    return max(found, key=lambda p: (lambda w, h: w * h)(*Image.open(p).size), default=None)


def collect_outputs(raw_dir: Path, output_dir: Path, images: list[Path], scale: int) -> dict:
    """결과를 <원본 이름>.png로 모읍니다. 크기가 정확히 scale배가 아니면 맞추고 그 사실을 기록합니다."""
    results = {}
    for image in images:
        found = find_output(raw_dir, image.stem)
        if found is None:
            results[image.stem] = {"status": "missing"}
            continue
        with Image.open(image) as source, Image.open(found) as result:
            want = (source.width * scale, source.height * scale)
            entry = {"status": "ok", "model_size": list(result.size)}
            if result.size != want:
                entry["resized_to"] = list(want)  # ponytail: 크기 맞춤은 Lanczos. 차이가 크면 보고서에서 확인
                result = result.resize(want, Image.LANCZOS)
            result.convert("RGB").save(output_dir / f"{image.stem}.png")
        results[image.stem] = entry
    return results


def run(name: str, input_dir: Path, output_dir: Path, scale: int) -> dict:
    spec = load_spec(name)
    if scale not in spec["run"]["scales"]:
        raise ValueError(f"{name}: 배율 {scale}은 지원하지 않습니다 (지원: {spec['run']['scales']})")
    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"{input_dir}에 사진이 없습니다")
    raw_dir = output_dir / RAW_FOLDER
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(name, spec, input_dir, raw_dir, scale, images)
    env = os.environ | {"HF_HUB_OFFLINE": "1"} | spec["run"].get("env", {})  # 실행 중 몰래 내려받지 않음
    seconds, peak_mb, code = watch(command, code_folder(name), env, output_dir / "run.log")
    info = {
        "model": name, "scale": scale, "images": len(images), "returncode": code,
        "seconds": round(seconds, 2), "seconds_per_image": round(seconds / len(images), 2),
        "peak_vram_mb": round(peak_mb), "outputs": collect_outputs(raw_dir, output_dir, images, scale) if code == 0 else {},
    }
    (output_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def main():
    parser = argparse.ArgumentParser(prog="python -m models.generative_ai.upscales.run_model", description="모델 하나로 업스케일합니다")
    parser.add_argument("model", help="모델 폴더 이름 (예: vosr)")
    parser.add_argument("--input", type=Path, required=True, help="입력 사진 폴더")
    parser.add_argument("--scale", type=int, default=4, help="배율")
    parser.add_argument("--output", type=Path, help="결과 폴더 (기본: settings.toml output_dir/<모델>)")
    args = parser.parse_args()
    output = args.output or project_path("output_dir") / args.model
    info = run(args.model, args.input, output, args.scale)
    shown = output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output
    print(json.dumps({k: v for k, v in info.items() if k != "outputs"}, ensure_ascii=False))
    print(f"결과: {shown} (실행 기록: run.log)")


if __name__ == "__main__":
    main()
