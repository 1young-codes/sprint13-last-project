# 자동 사진 보정

카메라 각도 단계가 만든 사진을 사장님 대신 보정합니다. 업스케일보다 먼저 돕니다.

```
카메라 각도 → 자동 보정(이 폴더) → 업스케일
```

보정 앱처럼 필터를 고르게 하지 않습니다. 사진마다 무엇이 문제인지 진단하고, 상품의 실제 색과 모양이 바뀌지 않았는지 검사한 뒤, 바꾼 이유를 한국어로 남깁니다.

## 하는 일

1. 상품 영역을 나눕니다. `BiRefNet`이 앞쪽 피사체를 0~1 마스크로 만듭니다.
2. VLM(`Qwen3-VL-8B`)이 업종(음식, 상품, 매장)을 판단하고, 사진의 문제와 보정 값, 이유를 냅니다.
3. 후보를 만듭니다. VLM 제안 그대로, 제안을 약하게 줄인 것, VLM 없이 사진을 재서 정한 기본 자동 보정입니다.
4. 실물 검사에서 상품 색이나 모양을 바꾼 후보를 떨어뜨리고, 남은 후보 중 VLM이 광고에 가장 맞는 것을 고릅니다. 모두 떨어지면 원본을 그대로 씁니다.
5. 배경 잡동사니를 지웁니다(`LaMa`). 상품 영역은 건드리지 않습니다.
6. 선택 기능으로 스튜디오 조명을 입힙니다(`IC-Light`). 상품의 작은 세부는 원본에서 다시 가져오고, 다시 검사해 통과할 때만 따로 저장합니다.

VLM은 숫자와 이유만 냅니다. 픽셀은 `adjustments.py`가 밝기와 색만 바꾸므로, 1~4단계에서는 사진 내용이 새로 그려지지 않습니다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `settings.toml` | 경로, 보정 값 범위, 실물 검사 기준, 모델 실행 값 |
| `categories.toml` | 업종별 설명, 값 범위, 검사 기준, 스튜디오 조명 문구 |
| `prompts.toml` | VLM에게 주는 지시문 |
| `model_sources.toml` | 모델 저장소와 고정 커밋 |
| `adjustments.py` | 보정 값 적용 (노출, 색온도, 대비, 채도 등 10가지) |
| `auto_params.py` | VLM 없이 쓰는 기본 자동 보정 |
| `fidelity.py` | 실물 검사: 색 차이(ΔE), 구조 유사도, 새로 날아간 화소 |
| `product_mask.py`, `vlm.py`, `cleanup.py`, `studio_light.py` | 모델 단계 |
| `retouch.py` | 전체 순서 |

## 준비

VM에서 한 번만 합니다. 공용 그래픽 패키지(`requirements-gpu.txt`) 위에 몇 가지를 더합니다.

```bash
uv venv --python 3.12 /data/venvs/photo_retouch
uv pip install --python /data/venvs/photo_retouch/bin/python -r models/generative_ai/photo_retouch/requirements.txt
HF_HOME=/data/models/hf_cache /data/venvs/photo_retouch/bin/python -m models.generative_ai.photo_retouch.download_models
```

모델은 약 25GB입니다. 결과 폴더는 `.env`의 `RETOUCH_OUTPUT_DIR`로 바꿀 수 있습니다(프로젝트 루트 기준).

## 실행

```bash
python -m models.generative_ai.photo_retouch 사진.jpg                    # 업종은 VLM이 판단
python -m models.generative_ai.photo_retouch 사진.jpg --category food --studio-light
python -m models.generative_ai.photo_retouch 사진.jpg --no-vlm           # 그래픽 메모리가 모자랄 때
```

사진마다 `outputs/photo_retouch/<사진 이름>/`에 아래 파일이 생깁니다.

| 파일 | 내용 |
| --- | --- |
| `retouched.png` | 보정 결과 |
| `studio.png` | 스튜디오 조명 결과 (선택, 검사 통과 시) |
| `compare.png` | 원본과 결과를 나란히 |
| `설명.md` | 사장님용 설명: 본 문제, 바꾼 값과 이유, 실물 검사 결과 |
| `report.json` | 전체 기록 |

## 조정

- 업종을 더하려면 `categories.toml`에 표 하나를 더합니다.
- 결과가 너무 약하거나 강하면 `settings.toml`의 `[candidates] strengths`와 `[ops]`를 바꿉니다.
- 실물 검사가 너무 엄격하면 업종별 `fidelity`를 넓힙니다.

## 사용 허락

| 모델 | 허락 |
| --- | --- |
| Qwen3-VL-8B-Instruct | Apache-2.0 |
| BiRefNet | MIT |
| big-lama | Apache-2.0 |
| IC-Light 코드 / Realistic Vision 5.1 | Apache-2.0 / creativeml-openrail-m |
