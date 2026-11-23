"""컨테이너에서 GPU, 모델 캐시, 결과 폴더가 이어지는지 확인하는 한 장 생성.

실행: docker compose run --rm gpu
결과: /data/outputs/demo/vm_smoke.png (화면은 이 파일을 쓰지 않는다)
"""
import os
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stabilityai/sd-turbo"  # 약 2.5GB, 1스텝 생성. 확인용이라 가벼운 모델을 쓴다
OUT = Path("/data/outputs/demo/vm_smoke.png")
PROMPT = "a fresh red sea bream on crushed ice, product photo, soft studio light"
SEED = 0

# 팀원(adteam 그룹)도 결과와 캐시 파일을 고치고 지울 수 있게 그룹 쓰기 권한을 남긴다
os.umask(0o002)

assert torch.cuda.is_available(), "컨테이너 안에서 GPU가 보이지 않습니다 (compose의 gpus: all 확인)"
print("GPU:", torch.cuda.get_device_name(0))

t0 = time.perf_counter()
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, variant="fp16"
).to("cuda")
load_s = time.perf_counter() - t0

torch.cuda.reset_peak_memory_stats()
t1 = time.perf_counter()
image = pipe(
    PROMPT,
    num_inference_steps=1,
    guidance_scale=0.0,  # sd-turbo는 CFG 없이 학습됨
    width=512,
    height=512,
    generator=torch.Generator("cuda").manual_seed(SEED),
).images[0]
torch.cuda.synchronize()
gen_s = time.perf_counter() - t1
peak_gb = torch.cuda.max_memory_allocated() / 1024**3

OUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUT)
assert OUT.exists() and image.size == (512, 512)

print(f"saved {OUT} {image.size}")
print(f"model load {load_s:.1f}s | generate {gen_s:.2f}s | peak VRAM {peak_gb:.2f}GB")
