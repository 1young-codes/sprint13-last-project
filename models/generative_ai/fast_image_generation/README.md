# 빠른 이미지 생성

Qwen-Image 2.1 편집 모델을 불러와 빠르게 돌리는 공용 코드입니다. 그래픽 카드 L4(24GB) 한 장에 맞춰 정수 8비트 가중치로 불러오고, 캐싱으로 생성 단계를 줄입니다. 이 모델로 이미지를 만드는 기능이라면 어디서든 가져다 씁니다.

## 폴더 구성

| 경로 | 하는 일 |
| --- | --- |
| `qwen_image_model.py` | 모델 불러오기와 편집 실행 (`edit`, `edit_with_reference`) |
| `int8_linear_layers.py` | 정수 8비트 가중치 층, LoRA 붙이고 떼기 |
| `model_files.py` | `model_sources.toml`에 적힌 모델을 허깅페이스 캐시에서 찾거나 내려받기 |
| `qwen_image_caching/` | 지금 쓰는 캐싱 두 가지와 그 설정, 검증 코드 |

## 쓰는 법

```bash
python -m models.generative_ai.fast_image_generation.model_files   # 모델 약 16GB, 팀 공용 캐시(HF_HOME)에 한 번만 받음
```

```python
from models.generative_ai.fast_image_generation.qwen_image_model import QwenImageModel

model = QwenImageModel(ratios_file=내_기능_폴더 / "mag_cache_ratios.json")
model.use_lora(None, 0)                      # LoRA를 쓰면 파일 경로와 세기를 넘김
result = model.edit(image, "프롬프트", seed=0, steps=25)
```

## 캐싱

`qwen_image_caching/cache_settings.toml`에서 켜고 끕니다.

| 설정 | 하는 일 | 결과 |
| --- | --- | --- |
| `reuse_condition_cache` | 글자와 참고 이미지 부분을 첫 단계에서 계산해 두고 재사용 | 같음 |
| `mag_cache` | 변화가 작은 단계는 계산을 건너뛰고 앞 단계의 변화량을 재사용 | 아주 조금 달라짐 |

카메라 앵글 기능에서 두 가지를 모두 켜니 프리셋 한 장이 85초에서 21~38초로 줄었습니다.

MagCache는 단계별 크기 비율이 있어야 켜집니다. 비율은 모델, LoRA, 단계 수에 따라 달라서 기능마다 자기 사진 여러 장으로 한 번 잽니다. 비율 파일이 없으면 모든 단계를 계산합니다.

```python
from models.generative_ai.fast_image_generation.qwen_image_caching.mag_cache import save_ratios

model.calibration = {}                       # 이 동안은 MagCache 없이 모든 단계를 계산하며 비율을 모음
for image in 성격이_다른_사진들:
    model.edit(image, "프롬프트", seed=0, steps=25)
save_ratios(내_기능_폴더 / "mag_cache_ratios.json", model.calibration, steps=25, photos=len(성격이_다른_사진들))
```

## 검증

```bash
python -m pytest models/generative_ai/fast_image_generation
```
