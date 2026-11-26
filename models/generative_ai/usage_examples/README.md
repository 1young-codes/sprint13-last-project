# 활용 예시 이미지

사장님이 올린 원본 사진 한 장으로, 그 상품이 어울리는 장면에 놓인 광고용 사진을 새로 만듭니다.

```
원본 사진 → 해석(GPT 두 번) → MOSS 생성 → 각도 → 업스케일 → 보정
```

## 하는 일

1. 인식: 해석 모델(`gpt-6-luna`)이 사진 속 상품을 생성 모델이 다시 그릴 수 있을 만큼 자세히 묘사합니다. 생선이면 종과 부위, 채소면 품종, 화장품이면 용기 모양과 색까지 적습니다.
2. 분류와 장면: 그 묘사만 보고 업종을 고르고, 업종별 장면 방향에 맞는 장면 후보 세 개를 영어 프롬프트로 냅니다. 첫 번째가 가장 어울리는 것입니다.
3. 생성: MOSS-Image SFT가 그 프롬프트로 그립니다. 참고 이미지는 쓰지 않습니다. 공식 모델이 참고 이미지를 받지 못하기 때문입니다.
4. 다듬기: 생성한 이미지를 기존 기능으로 각도를 바꾸고, 키우고, 보정합니다.

업종은 보정 기능과 같은 표(`../photo_retouch/categories.toml`)를 씁니다. 음식, 화장품, 생활용품·가전, 상품, 매장이 있고, 업종마다 `example_scene`에 장면 방향이 적혀 있습니다.

음식은 음식 자체를 어울리는 배경에 연출합니다. 완성 요리는 같은 요리로 두고, 원재료는 그 재료로 만든 요리로 바꾸되 종과 부위는 그대로 둡니다. 예를 들어 고등어 원물 사진은 고등어조림 같은 요리가 됩니다.

## 생성 설정

| 항목 | 값 | 근거 |
| --- | --- | --- |
| 모델 | MOSS-Image SFT, bf16 | 공식 저장소 고정 커밋 |
| 단계, CFG | 50, 3 | 공식 기본값. 부정 갈래 22번 블록 건너뛰기도 기본값 |
| 크기 | 가로 1216×832, 세로 832×1216 | SDXL 공식 해상도 중 약 3:2 |
| 시드 | 사진마다 다르고, 세 조건은 같음 | 차이가 로라에서만 나오게 |

비교 조건은 세 가지입니다. 프롬프트와 시드는 같고, 로라2만 배포 페이지대로 프롬프트 맨 앞에 트리거 단어 `cndamiph`를 붙입니다.

| 조건 | 로라 파일 | 세기 | 형식 |
| --- | --- | --- | --- |
| 오리지널 | 없음 | | |
| 로라1 | `MOSSRealisticSnapshotV1.safetensors` | 1.0 | LoKr, q·k·v 묶음 |
| 로라2 | `cndamiph_Candid_Amateur_iPhone_MOSS_v1.safetensors` | 1.0 | LoRA |

로라2 배포 페이지에는 MOSS Base로 학습했다고 적혀 있습니다. 이번 비교는 SFT로 정했으므로, 로라2 효과가 학습할 때보다 약할 수 있습니다.

## 생성 속도

공용 폴더 `../fast_image_generation/moss_image_model.py`가 두 가지 캐싱을 씁니다.

| 캐싱 | 하는 일 | 결과 |
| --- | --- | --- |
| 인코딩 저장 | 프롬프트를 문장 인코더로 한 번만 바꿔 파일로 두고 다시 씀. 빈 부정 프롬프트는 모든 이미지가 함께 씀 | 같음 |
| MagCache | 변화가 작은 단계는 블록 계산을 건너뛰고 앞 단계의 변화량을 다시 씀 | 아주 조금 달라짐 |

Qwen 쪽에서 쓰던, 블록 안의 글 부분 계산을 저장하는 조건 캐시는 쓰지 않습니다. MOSS는 글 토큰도 매 단계 그림 토큰과 같은 블록을 지나며 단계마다 다른 조정값을 받기 때문에, 저장해 두면 결과가 바뀝니다.

MagCache 비율은 조건마다 다릅니다. 로라가 생성 모델의 변화량을 바꾸기 때문입니다. CFG 때문에 한 단계에 긍정과 부정 두 번 도는데, 갈래마다 상태를 따로 둡니다. 비율은 `mag_cache_ratios.json`에 있고, 모델이나 로라, 단계 수를 바꾸면 `calibrate`로 다시 잽니다.

## 가상 머신 준비

L4(24GB)와 시스템 메모리 15GB에 맞춰, 문장 인코더(약 17GB)로 프롬프트를 모두 인코딩해 내린 다음 생성 모델(약 10GB)을 올립니다.

```bash
source /data/venv/bin/activate
python -m models.generative_ai.fast_image_generation.moss_image_model   # 모델 약 27GB, HF_HOME 캐시에 한 번만
```

로라 두 개는 배포 페이지에서 받아 `/data/models/loras`에 둡니다. 파일 해시는 `settings.toml`에 있고, 다르면 생성 전에 멈춥니다. 프로젝트 루트 `.env`에 아래 값을 둡니다.

| 키 | 가상 머신 값 |
| --- | --- |
| `OPENAI_API_KEY` | 해석 모델 키 |
| `USAGE_EXAMPLES_LORA_DIR` | `/data/models/loras` |
| `USAGE_EXAMPLES_OUTPUT_DIR` | `/data/outputs/usage_examples` |
| `USAGE_EXAMPLES_RETOUCH_PYTHON` | `/data/venvs/photo_retouch/bin/python` |

각도, 업스케일, 보정은 각 폴더 설명서대로 먼저 준비해 둡니다.

## 실행

사진 한 장을 끝까지 만듭니다. 결과는 `<출력 폴더>/<사진 이름>/`에 쌓입니다.

```bash
python -m models.generative_ai.usage_examples run --input 사진.jpg
```

로라 비교 실험은 단계별로 돌립니다. 폴더 하나가 실험 하나이고, 끊겨도 다시 실행하면 이어서 합니다.

```bash
R=/data/outputs/usage_examples/lora-compare
python -m models.generative_ai.usage_examples plan --input photos --run $R           # 해석
python -m models.generative_ai.usage_examples calibrate --run $R --category food     # MagCache 비율
python -m models.generative_ai.usage_examples generate --run $R --category food      # 30장 × 3조건
python -m models.generative_ai.usage_examples generate --run $R --category cosmetic household --condition original
python -m models.generative_ai.usage_examples finish --run $R --id <사진 id>          # 한 장만 각도 → 업스케일 → 보정
python -m models.generative_ai.usage_examples review --run $R --out 평가폴더
python -m models.generative_ai.usage_examples summarize 순위_*.csv
```

## 사람 평가

`review`가 만든 폴더의 `index.html`을 브라우저로 엽니다. 인터넷 없이 됩니다. 음식마다 원본 사진과 세 조건의 결과가 나란히 보이고, 조건 이름과 로라 파일 이름이 함께 적혀 있습니다. 사진을 누르면 크게 보입니다.

기준은 하나입니다: AI 같은 질감을 얼마나 걷어냈나. 세 장에 1위(가장 실제 사진 같음)부터 3위까지 매기고 `CSV 저장`을 누르면 `순위_<이름>.csv`가 내려받아집니다. 매긴 순위는 브라우저에 남아서 닫았다 열어도 이어서 매길 수 있습니다. 평가자들의 파일을 `summarize`로 합치면 조건별 평균 순위와 1위 횟수가 나옵니다.

## 파일

| 파일 | 하는 일 |
| --- | --- |
| `settings.toml` | 경로, 해석 모델, 생성 설정, 비교 조건, 다듬기 단계 |
| `prompts.toml` | 해석 모델 지시문 (인식, 분류와 장면) |
| `plan.py` | 해석 두 번과 답 다듬기 |
| `generate.py` | 조건별 생성, 로라 해시 확인, 시간과 그래픽 메모리 기록 |
| `finish.py` | 각도, 업스케일, 보정을 각자의 명령으로 이어 실행 |
| `review_page.py` | 평가 폴더 만들기, 평가 결과 합치기 |
| `photos/sources.toml` | 실험 원본 사진의 출처, 작가, 이용 허락 |
| `tests/` | 그래픽 카드 없이 도는 검증 |
