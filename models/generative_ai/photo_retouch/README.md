# 자동 사진 보정

카메라 각도 단계가 만들고 업스케일로 키운 사진을 사장님 대신 보정합니다.

```
카메라 각도 → 업스케일 → 자동 보정(이 폴더)
```

보정 앱처럼 필터를 씌우지 않습니다. 사진마다 잰 문제만 바로잡고 업종별로 아주 약한 색감 하나만 더하며, 상품의 실제 색과 모양이 바뀌지 않았는지 검사한 뒤, 바꾼 이유를 한국어로 남깁니다.

## 하는 일

1. 상품 영역을 나눕니다. `BiRefNet`이 앞쪽 피사체를 0~1 마스크로 만듭니다.
2. 사진을 재서 바로잡을 거리를 찾고 값을 계산합니다: 밝기, 색 치우침, 대비, 밝은 곳 날아감, 어두운 곳 뭉개짐.
3. 해석 모델(OpenAI, 기본 `gpt-5.4-mini`)이 업종을 판단하고, 잰 문제마다 진짜 문제인지 일부러 낸 분위기인지 판단해 이유를 씁니다. 식당의 따뜻한 조명처럼 분위기로 본 것은 그대로 둡니다.
4. 후보를 만듭니다. 바로잡기에 업종 색감(`categories.toml`의 `look`)을 더한 것, 그걸 약하게 줄인 것, 판단 없이 잰 것을 전부 바로잡은 것입니다.
5. 실물 검사에서 상품 색이나 모양을 바꾼 후보를 떨어뜨리고, 남은 후보 중 해석 모델이 광고에 가장 맞는 것을 고릅니다. 모두 떨어지면 원본을 그대로 씁니다.
6. 선택 기능으로 배경 잡동사니를 지웁니다(`LaMa`, `--cleanup`). 상품 영역은 건드리지 않습니다. 해석 모델이 짚는 물건 위치가 자주 어긋나서 기본은 꺼 두었습니다.
7. 선택 기능으로 스튜디오 조명을 입힙니다(`IC-Light`). 상품은 원본 색과 세부를 두고 새 조명의 밝기만 입히며, 다시 검사해 통과할 때만 따로 저장합니다.

해석 모델은 판단과 이유만 냅니다. 값은 사진을 재서 정하고, 픽셀은 `adjustments.py`가 밝기와 색만 바꾸므로 1~5단계에서는 사진 내용이 새로 그려지지 않습니다.

윤곽·선명도 조정은 쓰지 않습니다. 경계에 테두리를 만들고, 업스케일이 이미 선명하게 해 주기 때문입니다. 작은 사진을 그대로 보정하면 원본 압축 무늬와 테두리가 도드라지므로, 업스케일을 먼저 하는 것이 좋습니다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `settings.toml` | 경로, 보정 값 범위, 실물 검사 기준, 모델 실행 값 |
| `categories.toml` | 업종별 설명, 값 범위, 검사 기준, 스튜디오 조명 문구 |
| `prompts.toml` | 해석 모델에게 주는 지시문 |
| `model_sources.toml` | 모델 저장소와 고정 커밋 |
| `adjustments.py` | 보정 값 적용 (노출, 색온도, 색조, 대비, 밝은 곳, 어두운 곳, 생동감, 채도) |
| `auto_params.py` | 바로잡을 거리 재기와 값 계산 |
| `fidelity.py` | 실물 검사: 색 차이(ΔE), 구조 유사도, 새로 날아간 화소 |
| `interpret.py` | 사진 해석 (OpenAI) |
| `product_mask.py`, `cleanup.py`, `studio_light.py` | 그래픽 카드에서 도는 모델 단계 |
| `retouch.py` | 전체 순서 |

## 준비

VM에서 한 번만 합니다. 공용 그래픽 패키지(`requirements-gpu.txt`) 위에 몇 가지를 더합니다.

```bash
uv venv --python 3.12 /data/venvs/photo_retouch
uv pip install --python /data/venvs/photo_retouch/bin/python -r models/generative_ai/photo_retouch/requirements.txt
HF_HOME=/data/models/hf_cache /data/venvs/photo_retouch/bin/python -m models.generative_ai.photo_retouch.download_models
```

그래픽 카드 모델은 약 8GB입니다. 프로젝트 루트 `.env`에 `OPENAI_API_KEY`가 있어야 해석이 돕니다. 결과 폴더는 `.env`의 `RETOUCH_OUTPUT_DIR`로 바꿀 수 있습니다(프로젝트 루트 기준).

## 실행

```bash
python -m models.generative_ai.photo_retouch 사진.jpg                    # 업종은 해석 모델이 판단
python -m models.generative_ai.photo_retouch 사진.jpg --category food --studio-light
python -m models.generative_ai.photo_retouch 사진.jpg --no-ai            # 해석 없이 기본 자동 보정만 (키가 없을 때)
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
- 업종 색감은 `categories.toml`의 `look`, 바로잡기 기준과 세기는 `settings.toml`의 `[auto]`에서 바꿉니다.
- 실물 검사가 너무 엄격하면 업종별 `fidelity`를 넓힙니다.

## 사용 허락

| 모델 | 허락 |
| --- | --- |
| BiRefNet | MIT |
| big-lama | Apache-2.0 |
| IC-Light 코드 / Realistic Vision 5.1 | Apache-2.0 / creativeml-openrail-m |
