"""장면 프롬프트로 MOSS-Image 이미지를 만듭니다 (그래픽 카드 필요).

작업(job)은 {"id", "prompt", "orientation", "seed"}입니다. 조건(오리지널, 로라1, 로라2)마다 모델을 한 번 불러와
모든 작업을 만들고 내립니다. 이미 만든 이미지는 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 만듭니다.
"""

import gc
import json
import subprocess
import time
from pathlib import Path

from .settings import RATIOS_FILE, condition_prompt, conditions, load_settings, project_path


def lora_file(condition: dict) -> Path | None:
    """조건의 로라 파일. 없거나 해시가 다르면 오류 (다른 버전 로라로 비교하지 않게)."""
    if "file" not in condition:
        return None
    from ..fast_image_generation.model_files import sha256_of

    path = project_path("lora_dir") / condition["file"]
    if not path.is_file():
        raise FileNotFoundError(f"{condition['label']} 로라 파일이 없습니다: {path} (내려받을 곳: {condition['page']})")
    if sha256_of(path) != condition["sha256"]:
        raise ValueError(f"{path.name}의 해시가 settings.toml과 다릅니다. 배포 페이지의 같은 버전인지 확인하세요")
    return path


def gpu_users() -> list[str]:
    """지금 그래픽 카드를 쓰는 다른 프로그램 (pid, 이름, 메모리). nvidia-smi가 없으면 빈 목록."""
    try:
        output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                 "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def require_free_gpu() -> None:
    """팀원이 그래픽 카드를 쓰는 중이면 멈춥니다. 같이 올리면 둘 다 메모리가 모자라 죽습니다 (L4 24GB)."""
    if users := gpu_users():
        raise RuntimeError("다른 프로그램이 그래픽 카드를 쓰는 중이라 멈춥니다. 끝난 뒤 다시 실행하세요:
  " + "
  ".join(users))


def load_check(condition_names: list[str] | None = None) -> dict:
    """추론 없이 모델이 그래픽 카드에 올라가는지만 봅니다: 문장 인코더, 그리고 조건마다 생성 모델과 로라 합치기.
    올렸다가 바로 내리며, 단계마다 걸린 시간과 그래픽 메모리를 돌려줍니다."""
    import torch

    from ..fast_image_generation.moss_image_model import MossImageModel, load_text_encoder

    require_free_gpu()
    chosen = conditions(condition_names)
    files = [lora_file(c) for c in chosen]
    report = {}

    def measure(name, load):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loaded = load()
        report[name] = {"load_seconds": round(time.perf_counter() - started, 1),
                        "vram_gb": round(torch.cuda.memory_allocated() / 2**30, 1),
                        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(f"[올라감] {name}: {report[name]}", flush=True)
        return loaded

    encoder = measure("문장 인코더", load_text_encoder)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    for condition, path in zip(chosen, files):
        model = measure(f"생성 모델 + {condition['label']}", lambda: MossImageModel(path, condition.get("strength", 1.0),
                                                                                  RATIOS_FILE))
        report[f"생성 모델 + {condition['label']}"]["merged_lora_layers"] = model.merged_layers
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return report


def generate(jobs: list[dict], output_dir: Path, condition_names: list[str] | None = None,
             calibrate: bool = False) -> dict:
    """output_dir/<조건>/<id>.png를 만듭니다. calibrate면 MagCache 없이 모든 단계를 계산하며 크기 비율을 재서 저장합니다."""
    import torch

    from ..fast_image_generation.moss_image_model import MossImageModel, encode_prompts
    from ..fast_image_generation.qwen_image_caching.mag_cache import save_ratios

    require_free_gpu()
    settings = load_settings()["generation"]
    chosen = conditions(condition_names)
    info_file = output_dir / "generate_info.json"
    files = [lora_file(c) for c in chosen]  # 오래 걸리는 생성 전에 로라부터 확인
    negative = settings["negative_prompt"]
    prompts = [condition_prompt(c, job["prompt"]) for c in chosen for job in jobs]
    embeddings = encode_prompts([negative, *prompts], project_path("output_dir") / "embeddings")
    # 이어서 만들 때 앞서 만든 조건의 기록을 지우지 않게 합칩니다
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.is_file() else {"conditions": {}}
    info |= {"steps": settings["steps"], "cfg": settings["cfg"]}
    measured = {}
    for condition, path in zip(chosen, files):
        folder = output_dir / condition["name"]
        folder.mkdir(parents=True, exist_ok=True)
        todo = [job for job in jobs if calibrate or not (folder / f"{job['id']}.png").is_file()]
        if not todo:
            continue
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = MossImageModel(path, condition.get("strength", 1.0), RATIOS_FILE)
        load_seconds, seconds = round(time.perf_counter() - started, 1), {}
        if calibrate:
            model.calibration = []
        for job in todo:
            width, height = settings["sizes"][job["orientation"]]
            started = time.perf_counter()
            image = model.generate(torch.load(embeddings[condition_prompt(condition, job["prompt"])]),
                                   torch.load(embeddings[negative]), width, height, job["seed"],
                                   settings["steps"], settings["cfg"])
            image.save(folder / f"{job['id']}.png")
            seconds[job["id"]] = round(time.perf_counter() - started, 1)
            print(f"[{condition['label']}] {job['id']} {seconds[job['id']]}초", flush=True)
        info["conditions"][condition["name"]] = {
            "label": condition["label"], "lora": condition.get("file"), "strength": condition.get("strength"),
            "merged_layers": model.merged_layers, "mag_cache": model.ratios is not None and not calibrate,
            "load_seconds": load_seconds, "seconds": seconds,
            "seconds_per_image": round(sum(seconds.values()) / len(seconds), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        if calibrate:
            measured[model.name] = model.calibration
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if calibrate and measured:
        save_ratios(RATIOS_FILE, measured, settings["steps"], len(jobs))
    info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info
