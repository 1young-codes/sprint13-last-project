# 레이어 광고 이미지

보정까지 끝난 제품 사진으로 광고 이미지를 만듭니다. 결과는 한 장짜리 그림이 아니라 레이어 네 종류로 나뉘어 있어서, 하나만 바꿀 수 있습니다.

```
카메라 각도 → 업스케일 → 자동 보정 → 레이어 광고(이 폴더)
```

| 레이어 | 파일 | 만드는 방법 | 바꿀 때 |
| --- | --- | --- | --- |
| 배경 | `background.png` | 빈 캔버스에서 제품과 글자 없이 생성 | 배경만 다시 생성 |
| 그림자·반사 | `shadow.png` | 제품 마스크를 바닥에 투영해 접지 그림자를 만듦 | 제품을 옮기면 같이 옮겨짐 |
| 제품 | `product.png` | 보정된 사진에서 잘라 낸 원본 화소 그대로 | 위치만 바꿈 |
| 글자 (줄마다) | `text_1_title.png` 등 | 고정 한글 폰트로 브라우저가 그림 | 문구만 고치고 다시 합침 |

위치, 문구, 폰트, 색은 `layout.json`에 있고, `ad.png`는 이 파일대로 레이어를 합친 결과입니다.

## 한국어 글자를 깨지지 않게 하는 방법

1. 제품 위 여백에 제목, 부제, 가격 자리를 두고 배경 밝기에 맞는 글자색을 고릅니다.
2. 언어모델이 제목과 부제 초안을 씁니다. 가격은 입력한 값 그대로 표시합니다.
3. 브라우저가 고정 한글 폰트로 줄마다 투명 레이어를 그립니다. `layout.json`의 문구나 위치를 고쳐 다시 합칠 수 있습니다.

## 분위기와 폰트

`settings.toml`의 `[moods]`에 분위기 세 가지가 있습니다. 폰트는 모두 SIL 오픈 폰트 라이선스라 상업 광고에 쓸 수 있습니다.

| 분위기 | 제목 폰트 |
| --- | --- |
| `minimal` 미니멀 스튜디오 | 본고딕 굵게 (Noto Sans KR) |
| `lifestyle` 생활 연출 | 본명조 (Noto Serif KR) |
| `sale` 계절·행사 세일 | 검은고딕 (Black Han Sans) |

폰트는 처음 쓸 때 `data/fonts/`로 내려받습니다.

## 준비

VM에서 한 번만 합니다. 보정 단계 환경을 그대로 쓰고, 글자를 그릴 브라우저만 더합니다.

```bash
uv pip install --python /data/venvs/photo_retouch/bin/python -r models/generative_ai/ad_layers/requirements.txt
/data/venvs/photo_retouch/bin/playwright install --with-deps chromium
```

생성 모델(Qwen-Image-2.1)은 `fast_image_generation` 폴더 README대로, 제품 마스크(BiRefNet)는 `photo_retouch` 폴더 README대로 받아 둡니다. 프로젝트 루트 `.env`에 `OPENAI_API_KEY`가 있어야 합니다.

## 실행

```bash
# 분위기 세 가지로 한 장씩
python -m models.generative_ai.ad_layers.make_ad 사진.png --name "딸기 생크림 케이크" --unit "1조각" --price 6500 \
    --usage "오후 커피와 함께" --idea "햇살 드는 카페 테이블"

# layout.json에서 문구나 위치를 고친 뒤, 생성 없이 다시 합치기
python -m models.generative_ai.ad_layers.make_ad --recompose outputs/ad_layers/사진/minimal
```

결과는 `outputs/ad_layers/<사진 이름>/<분위기>/`에 저장됩니다. `base.png`는 글자를 얹기 전 합성입니다. 단계별 걸린 시간은 `layout.json`의 `seconds`에 있습니다.

## 포스터 구도를 먼저 정하기

`scene_ad`는 상품 사진과 참고 포스터를 보고 강조점, 사진 영역, 문구별 배치를 정합니다. 그 구도를 바탕으로 피사체의 위치, 조명, 글자용 여백을 촬영 지시로 작성합니다. 원래 배경을 제거한 상품을 계획한 위치와 크기에 놓아 Qwen의 참고 사진으로 전달합니다. Qwen이 새 장면을 만들면 글자나 색면이 실제 피사체를 가릴 때만 배치를 조정합니다. 조정이 적용되면 처음 배치한 포스터를 `before-typesetting.png`로 함께 저장합니다. 두 포스터를 원래 크기로 열어 상품의 형태와 문구가 한 화면에서 자연스럽게 읽히는지 비교할 수 있습니다.

```bash
# 원본 사진을 넣은 구도 시안. OpenAI 호출과 글자 렌더링을 사용합니다.
python -m models.generative_ai.ad_layers.scene_ad photos/items.json photos outputs/poster_previews --preview --references 참고.png
python -m models.generative_ai.ad_layers.scene_ad photos/items.json photos outputs/scene_ads \
    --designs outputs/poster_previews --references 참고.png
python -m models.generative_ai.ad_layers.make_ad --recompose outputs/scene_ads/01_사진
```

입력 목록은 아래 모델 비교의 상품 목록 형식과 같습니다. 결과에는 `background.png`, 정해진 영역에 놓이는 `scene.png`, 제목·상품명·가격 레이어, `layout.json`, 합친 `ad.png`가 저장됩니다. 사진 위에 색면을 사용하면 `graphic.png`에 별도 레이어로 저장합니다. `layout.json`에서 문구와 위치를 수정한 뒤 글자 레이어를 다시 그릴 수 있습니다. 텍스트 레이어의 `break_after`에는 문구를 두 줄로 나눌 때 첫 줄에 들어가는 단어 수가 저장됩니다. 제품을 장면에서 분리해 이동할 때는 위의 `make_ad` 실행 경로를 사용합니다.

### 로컬에서 구도 반복 검토하기

WSL 2에서 `HF_HOME`을 SSD에 있는 리눅스 파일시스템 경로로 지정합니다. 모델을 받은 뒤, 저장된 시안을 다시 생성할 때 `--low-vram`을 붙이면 필요한 가중치 블록만 그래픽 카드로 옮깁니다.

```bash
export HF_HOME="$HOME/.local/share/ad-layers/hf_cache"
python -m models.generative_ai.fast_image_generation.model_files
python -m models.generative_ai.photo_retouch.download_models product_mask
python -m models.generative_ai.ad_layers.scene_ad photos/items.json photos outputs/local_ads \
    --designs outputs/poster_previews --low-vram
python -m models.generative_ai.ad_layers.scene_ad photos/items.json photos outputs/revised_ads \
    --revise outputs/local_ads --references 참고.png --low-vram
```

`--revise`는 기존 완성본과 원본 상품 사진을 보고 다음 연출에서 바꿀 점을 정합니다. 다음 기획은 상품 메시지와 사용자가 요청한 배경을 바탕으로 새 구도를 정합니다. 그 구도에서 제품의 위치와 글자에 필요한 여백을 촬영 지시로 만들어 Qwen에 전달합니다. `--rejected`에 보류한 후보를 전달하면 그 이미지도 다음 기획에서 참고합니다. 직접 보고 기록한 개선점이 있으면 각 후보의 `layout.json`에서 `revision.visual_review`에 적을 수 있습니다. 새 결과는 별도 폴더에 저장되므로 후보를 원래 크기에서 비교해 다음 개선에 사용할 완성본을 고를 수 있습니다.

도커로 실행할 때는 `docker build --target ad-layers -t sprint13-ad-layers .`로 브라우저까지 설치한 이미지를 만듭니다. 컨테이너에 `--gpus all`을 주고, SSD의 캐시를 `/data/models/hf_cache`에 연결해 같은 가중치를 사용합니다.

## 아직 하지 않은 것

- 다른 크기(4:5, 9:16, 쇼핑몰 썸네일, 16:9)로 바꾸기. 지금은 정사각형 1024×1024 한 가지입니다.
- 서버와 화면 연결.

## 참고 사진으로 광고 한 장을 통째로 생성해 비교하기

먼저 실행한 Qwen-Image-2.1 결과의 `settings.json`에서 파일명, 실제 상품명과 가격, 프롬프트를 읽습니다. MOSS와 FLUX는 같은 참고 사진을 받아 광고 전체를 다시 그립니다.

| 모델 | 실행 흐름 |
| --- | --- |
| MOSS-Image-SFT | 편집용 LoRA와 참고 사진을 양쪽 조건에 사용합니다. 원본 비율로 약 100만 화소를 만들고, 공개 예제의 Euler/normal 30단계, CFG 4와 22번 층 생략을 적용합니다. 이 실행기는 공식 BF16 기본 모델과 인코더를 사용합니다. [공개 워크플로](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_edit.json), [LoRA 제작자 코드](https://github.com/Chenkin-x/Moss-Image-Edit) |
| FLUX.3 [klein] | 공식 단독 실행 코드로 참고 사진을 넣습니다. FP8 증류 모델의 기본 16단계를 쓰고, 프롬프트 자동 확장은 끕니다. [공식 구현](https://github.com/black-forest-labs/flux-3-image-klein) |

GPU 파이썬 환경의 `PYTHONPATH`에 MOSS 편집 코드의 저장소 루트와 FLUX 공식 코드의 `src`를 추가합니다. 이 실험에 사용한 소스 커밋은 각각 `d735569`, `bfc8889`입니다.

가중치는 아래 구조에 둡니다. 가중치·토크나이저 파일명과 디렉터리는 배포본을 따릅니다.

| 경로 | 배포본 |
| --- | --- |
| `weights/moss/` | `chinoll/Moss-Image-Edit`의 어댑터 파일 두 개. 기본 모델 `OpenMOSS-Team/MOSS-Image-SFT`의 스냅샷은 별도로 지정합니다. |
| `weights/flux/` | `Runware/flux-3-image-klein`의 `variants/fp8r/`, `text_encoder/`, `ae.safetensors`. |

같은 결과 폴더로 다시 실행하면 이미 생성한 PNG를 건너뜁니다. 모델마다 결과 폴더를 따로 지정합니다.

상품마다 OpenAI `gpt-6-luna`가 네 단계를 순서대로 거쳐 캠페인 지시 하나를 만듭니다.

1. 기획 담당은 사진과 상품 정보로 메시지 하나와 그 근거를 정합니다.
2. 연출 담당은 같은 사진과 앞 단계 메시지를 받아 카메라, 배경, 조명을 정합니다.
3. 문구 담당은 앞에서 정한 메시지와 장면을 받아 공백 포함 24자 이내로 한글 제목을 씁니다.
4. 레이아웃 담당은 장면과 실제 문구를 받아 제품·제목·상품명·가격의 영역과 읽는 순서를 정합니다.

실행기가 입력한 상품명과 가격을 반영합니다. 배치 영역이 겹치거나 캔버스를 벗어나는 등 검사를 통과하지 못하면 레이아웃 담당에게 한 번 더 요청합니다. 단계별 결과와 생성 지시는 GPU 없이 JSON으로 저장하며, 같은 지시를 모델 비교에 사용할 수 있습니다.

이미지 모델에는 장면, 글자 스타일, 상대적인 배치 위치를 문장으로 전달합니다. 이미지에 들어갈 문구는 제목, 상품명, 가격 세 항목을 따옴표로 표시합니다. 기획 메시지와 배치 좌표는 단계별 결과에 저장합니다.

```bash
python -m models.generative_ai.ad_layers.creative_brief photos/items.json photos comparison/briefs.json
python -m models.generative_ai.ad_layers.compare_reference_ads flux comparison/briefs.json photos comparison/flux-brief weights
```

상품 목록의 각 항목은 참고 사진 파일명 `file`, 사장님의 한 줄 `idea`, 상품 정보 `fields`를 담습니다. `fields`에는 `product_name`, 원 단위 정수 `price_krw`와 선택 항목 `unit`, `usage_example`, `notes`를 넣습니다.

```bash
python -m models.generative_ai.ad_layers.compare_reference_ads moss qwen/settings.json photos comparison/moss weights \
    --moss-snapshot /path/to/MOSS-Image-SFT/snapshot
python -m models.generative_ai.ad_layers.compare_reference_ads flux qwen/settings.json photos comparison/flux weights
```

## 확인

```bash
python -m pytest models/generative_ai/ad_layers/tests
```

그래픽 카드 없이 제품 놓기, 접지 그림자, 글자 배치, 문구 다듬기, 합치기를 확인합니다.
