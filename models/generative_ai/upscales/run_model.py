"""모델 하나로 폴더 안 사진을 업스케일하고, 걸린 시간과 최대 그래픽 메모리를 잽니다.

    python -m models.generative_ai.upscales.run_model pixrestore --input 사진폴더 --scale 2

모델은 자기 가상환경의 파이썬으로 따로 실행되고(setup.toml [run] command), 이 모듈은 바깥에서 지켜봅니다.
결과는 모델마다 이름 규칙이 달라서, 원본 이름 그대로 <출력 폴더>/<원본 이름>.png로 모아 둡니다.
"""

import argparse
import json
import math
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


def build_command(name: str, spec: dict, input_dir: Path, output_dir: Path, scale: int) -> list[str]:
    values = {"input_dir": input_dir.resolve(), "output_dir": output_dir.resolve(), "scale": scale}
    return [str(venv_python(name)), *(part.format(**values) for part in spec["run"]["command"])]


def gpu_memory_by_pid() -> dict[int, float]:
    """nvidia-smi로 읽은 프로세스별 그래픽 메모리(MB). 따로 패키지를 깔지 않으려고 명령으로 읽습니다."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    rows = (line.split(",") for line in out.splitlines() if line.strip())
    return {int(pid): float(mb) for pid, mb in rows if mb.strip().isdigit()}


def watch(command: list[str], cwd: Path, env: dict, log_path: Path) -> tuple[float, float, int]:
    """명령을 실행하며 그 프로세스와 자식 프로세스의 그래픽 메모리 합의 최대값(MB)을 잽니다."""
    import psutil  # 프로젝트 GPU 환경에 accelerate와 함께 들어 있음

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
            peak = max(peak, sum(mb for pid, mb in gpu_memory_by_pid().items() if pid in pids))
            time.sleep(interval)
    return time.perf_counter() - start, peak, process.returncode


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


def tile_starts(length: int, size: int, overlap: int) -> list[int]:
    """size 조각이 적어도 overlap만큼 겹치며 length를 덮는 시작 위치. 조각 사이 간격은 고르게 나눕니다."""
    if length <= size:
        return [0]
    count = math.ceil((length - overlap) / (size - overlap))
    return [round(i * (length - size) / (count - 1)) for i in range(count)]


def cut_tiles(images: list[Path], scale: int, tiles: dict, tile_dir: Path) -> dict:
    """사진을 바이큐빅으로 scale배 키운 뒤 size×size 조각으로 저장합니다. 사진 이름 -> (크기, 조각 위치들)."""
    size, overlap = tiles["size"], tiles["overlap"]
    tile_dir.mkdir(parents=True, exist_ok=True)
    layouts = {}
    for path in images:
        with Image.open(path) as image:
            big = image.convert("RGB").resize((image.width * scale, image.height * scale), Image.BICUBIC)
        if min(big.size) < size:
            raise ValueError(f"{path.name}: 키운 크기 {big.size}가 조각 크기 {size}보다 작습니다")
        spots = [(x, y) for y in tile_starts(big.height, size, overlap) for x in tile_starts(big.width, size, overlap)]
        for x, y in spots:
            big.crop((x, y, x + size, y + size)).save(tile_dir / f"{path.stem}__x{x}_y{y}.png")
        layouts[path.stem] = (big.size, spots)
    return layouts


def join_tiles(layouts: dict, tiles: dict, tile_out: Path, raw_dir: Path) -> None:
    """조각 결과를 가장자리로 갈수록 작아지는 가중치로 섞어 이어 붙입니다 (VARestorer 타일 방식과 같은 원리)."""
    import numpy as np

    size = tiles["size"]
    ramp = np.minimum(np.arange(size) + 1, np.arange(size)[::-1] + 1).astype(np.float32)
    weight = np.minimum.outer(ramp, ramp)[..., None]
    for stem, ((width, height), spots) in layouts.items():
        total = np.zeros((height, width, 3), np.float32)
        count = np.zeros((height, width, 1), np.float32)
        for x, y in spots:
            found = next(tile_out.rglob(f"{stem}__x{x}_y{y}.png"), None)
            if found is None:
                raise FileNotFoundError(f"조각 결과가 없습니다: {stem}__x{x}_y{y}")
            with Image.open(found) as piece:
                total[y:y + size, x:x + size] += np.asarray(piece.convert("RGB").resize((size, size)), np.float32) * weight
            count[y:y + size, x:x + size] += weight
        Image.fromarray(np.clip(np.rint(total / count), 0, 255).astype(np.uint8)).save(raw_dir / f"{stem}.png")


def run(name: str, input_dir: Path, output_dir: Path, scale: int) -> dict:
    spec = load_spec(name)
    if scale not in spec["run"]["scales"]:
        raise ValueError(f"{name}: 배율 {scale}은 지원하지 않습니다 (지원: {spec['run']['scales']})")
    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"{input_dir}에 사진이 없습니다")
    raw_dir = output_dir / RAW_FOLDER
    raw_dir.mkdir(parents=True, exist_ok=True)
    tiles = spec["run"].get("tiles")
    env = os.environ | {"HF_HUB_OFFLINE": "1"} | spec["run"].get("env", {})  # 실행 중 몰래 내려받지 않음
    if tiles:  # 학습 해상도 조각으로만 제대로 도는 1배 복원 모델: 먼저 키우고 조각내서 돌린 뒤 이어 붙임
        tile_dir, tile_out = output_dir / "tiles", output_dir / "tiles_out"
        layouts = cut_tiles(images, scale, tiles, tile_dir)
        command = build_command(name, spec, tile_dir, tile_out, 1)
        seconds, peak_mb, code = watch(command, code_folder(name), env, output_dir / "run.log")
        if code == 0:
            join_tiles(layouts, tiles, tile_out, raw_dir)
    else:
        command = build_command(name, spec, input_dir, raw_dir, scale)
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
    parser.add_argument("model", help="모델 폴더 이름 (예: pixrestore)")
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
