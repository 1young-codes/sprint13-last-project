# 레이어 광고 이미지

보정까지 끝난 제품 사진으로 광고 이미지를 만듭니다. 결과는 한 장짜리 그림이 아니라 레이어 네 종류로 나뉘어 있어서, 하나만 바꿀 수 있습니다.

```
카메라 각도 → 업스케일 → 자동 보정 → 레이어 광고(이 폴더)
```

| 레이어 | 파일 | 만드는 방법 | 바꿀 때 |
| --- | --- | --- | --- |
| 배경 | `background.png` | 생성 모델이 글자 없이 그린 장면에서 제품과 둘레를 지우고 채움 | 배경만 다시 생성 |
| 그림자·반사 | `shadow.png` | 제품이 있는 장면과 빈 배경의 밝기 차이 | 제품을 옮기면 같이 옮겨짐 |
| 제품 | `product.png` | 보정된 사진에서 잘라 낸 원본 화소 그대로 | 위치만 바꿈 |
| 글자 (줄마다) | `text_1_title.png` 등 | 고정 한글 폰트로 브라우저가 그림 | 문구만 고치고 다시 합침 |

위치, 문구, 폰트, 색은 `layout.json`에 있고, `ad.png`는 이 파일대로 레이어를 합친 결과입니다.

## 한국어 글자를 깨지지 않게 하는 방법

생성 모델이 그린 글자는 쓰지 않습니다.

1. 배경, 그림자, 제품을 합친 그림 위에 생성 모델이 가짜 광고 글자를 그립니다(`draft.png`).
2. 초안과 원래 그림이 달라진 곳을 묶어 글자 줄마다 위치, 크기, 색을 읽습니다. 글자 인식 모델은 쓰지 않습니다.
3. 언어모델이 줄마다 제목, 부제, 가격, 행동 유도 중 역할을 정하고, 그 자리 길이에 맞는 문구를 씁니다. 가격은 입력한 값을 그대로 씁니다.
4. 진짜 한글을 고정 폰트로 그 자리에 그립니다.

배경은 처음부터 글자 없이 만들고 초안은 자리만 읽은 뒤 버리므로, 최종 결과에 생성 모델이 쓴 글자가 남지 않습니다.

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

결과는 `outputs/ad_layers/<사진 이름>/<분위기>/`에 저장됩니다. `scene.png`(제품이 들어간 장면)와 `draft.png`(가짜 글자 초안)는 확인용입니다. 단계별 걸린 시간은 `layout.json`의 `seconds`에 있습니다.

## 아직 하지 않은 것

- 다른 크기(4:5, 9:16, 쇼핑몰 썸네일, 16:9)로 바꾸기. 지금은 정사각형 1024×1024 한 가지입니다.
- 서버와 화면 연결.
- 네 모델(Qwen-Image-2.1, MOSS-Image, FLUX.3 Klein, Ideogram 4.0) 비교.

## 확인

```bash
python -m pytest models/generative_ai/ad_layers/tests
```

그래픽 카드 없이 제품 놓기, 그림자 분리, 글자 자리 찾기, 문구 다듬기, 합치기를 확인합니다.
