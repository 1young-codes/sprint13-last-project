# 업스케일

카메라 앵글을 바꾼 사진을 크게 키우는 단계입니다. 흐름은 각도 프리셋, 업스케일, 보정 순서입니다.
모델은 `PixRestore` XL 하나만 씁니다.

## 왜 이 모델인가

후보 다섯 개(FiDeSR, SeedVR2, VOSR, PixRestore, VARestorer)를 같은 사진으로 비교해, 실사를 가장 잘 지키는 모델을 골랐습니다.
비교 결과와 확대 그림은 이 기능을 추가한 변경 요청(#10)에 있습니다. 비교 코드와 다른 모델은 고른 뒤 지웠고, 필요하면 그 변경 요청의 커밋에서 되살릴 수 있습니다.

| 모델 | 결과 |
| --- | --- |
| PixRestore | 선정. 실제 흐름(×2)에서 정답에 가장 가깝고 외곽 번짐이 없음. 그래픽 메모리 4.3GB |
| VOSR | ×4는 충실하지만 ×2에서 외곽에 흰 테두리가 생김 |
| SeedVR2 | 가장 선명하지만 시계 글자 같은 없던 디테일을 지어냄 |
| FiDeSR | 디테일을 많이 지어내고, 짧은 변 768픽셀 미만 결과에서 모델 코드 오류 |
| VARestorer | 결과가 노이즈로 나옴 (설치 문제로 보였으나 고르지 않아 더 보지 않음) |

## 동작 방식

PixRestore는 1배 복원 모델입니다. 그래서 먼저 바이큐빅으로 키운 뒤 복원합니다.
학습 해상도가 512라서 큰 사진을 통째로 넣으면 8픽셀 격자 무늬로 무너집니다. 키운 사진을 512 조각으로 나눠 64픽셀씩 겹쳐 돌린 뒤 이어 붙입니다.
자세한 이유는 `pixrestore/setup.toml` 주석에 있습니다.

## 준비

가상환경은 `uv`로 만듭니다. 이미 끝난 단계는 건너뛰므로 여러 번 실행해도 됩니다.

```bash
python -m models.generative_ai.upscales.setup_model
```

모델은 자기 가상환경에서 돌고, 실행 스크립트는 프로젝트 GPU 환경(`requirements-gpu.txt`)에서 돌며 바깥에서 시간과 그래픽 메모리를 잽니다.
모델 환경은 `pixrestore/setup.toml`을 따라 다음을 준비합니다.

| 항목 | 내용 |
| --- | --- |
| `[environment]` | 파이썬 버전과 `requirements.txt`로 가상환경 생성 |
| `[code]` | 깃허브 추론 코드를 고정 커밋으로 받음 |
| `[weights.*]` | 허깅페이스 캐시(`HF_HOME`)에 고정 커밋으로 받음. DINOv2는 VOSR 저장소에 공개된 사본을 씀 |
| `[links]` | 추론 코드가 가중치를 찾는 자리에 캐시 파일을 링크로 연결 |

경로는 `settings.toml`에 프로젝트 루트 기준으로 적혀 있습니다. 가상 머신에서는 `.env`로 공용 폴더를 가리킵니다.

| 환경 변수 | 뜻 | 가상 머신 값 |
| --- | --- | --- |
| `UPSCALE_CODE_DIR` | 추론 코드 | `/data/models/vendor/upscales` |
| `UPSCALE_VENV_DIR` | 가상환경 | `/data/venvs/upscales` |
| `UPSCALE_OUTPUT_DIR` | 결과 | `/data/outputs/upscales` |

## 실행

폴더 안 사진을 키웁니다. 걸린 시간과 최대 그래픽 메모리를 `run_info.json`에 남깁니다.
프로젝트 GPU 환경의 파이썬으로 실행합니다. 가상 머신에서는 `/data/venv/bin/python`입니다.

```bash
python -m models.generative_ai.upscales.run_model pixrestore --input 사진폴더 --scale 2
```

L4에서 1024×683 사진을 ×2로 키우는 데 한 장에 약 98초 걸립니다. 사진마다 모델을 새로 읽기 때문입니다.

### 모델을 한 번만 불러 두고 쓰기

상주 서버나 여러 장을 이어서 처리할 때는 `upscale_photo.py`를 씁니다. 추론 코드를 같은 프로세스에서 불러 쓰므로 모델 가상환경의 파이썬으로 실행합니다.

```bash
/data/venvs/upscales/pixrestore/bin/python -m models.generative_ai.upscales.upscale_photo 사진.jpg 사진2.jpg --scale 2
```

```python
from models.generative_ai.upscales.upscale_photo import load_models, unload_models, upscale

models = load_models()                     # 모델을 올리고 첫 실행 준비까지 끝냄
big = upscale(photo, models, scale=2)      # photo: PIL 이미지. 여러 번 불러도 다시 읽지 않음
unload_models(models)                      # 그래픽 메모리 비우기
```

| 항목 | 사진마다 새로 실행 (`run_model`) | 한 번 불러 두기 (`upscale_photo`) |
| --- | --- | --- |
| 모델 불러오기 | 사진마다 | 처음 한 번 약 82초 (첫 실행 준비 포함) |
| 1024×683 사진 ×2 | 약 91초 | 약 5.7초 |
| 그래픽 메모리 | 약 4.1기가바이트 | 같음 |

첫 실행 때 추론 코드 일부(`torch.compile`)를 번역하는 데 약 1분이 들어서, 불러올 때 미리 끝내 둡니다.
두 방식의 결과는 조각마다 쓰는 난수 순서만 달라, 화소 차이가 최대 3/255입니다.

## 검증

그래픽 카드 없이 설정과 계산만 확인합니다.

```bash
python -m pytest models/generative_ai/upscales
```

## 라이선스

PixRestore 코드와 가중치는 Apache-2.0입니다.
