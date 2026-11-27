# 화면과 API가 주고받는 값

React 화면이 게시글을 만들 때 부르는 API의 주소와 값입니다. 화면은 브라우저에서 돌아서 VM 폴더를 직접 읽을 수 없으므로, 사진과 문구는 모두 이 API로 주고받습니다.

흐름은 `frontend/src/api/postApi.js`에 적어 둔 그대로입니다.

1. 사진과 입력을 보내면 작업 번호(`job_id`)를 받습니다.
2. 작업 번호로 상태를 2초마다 묻습니다. `ready`나 `failed`가 되면 멈춥니다.
3. `ready`면 결과의 이미지 주소로 사진을 받아 보여줍니다.

## 주소

| 방법 | 주소 | 하는 일 |
| --- | --- | --- |
| `POST` | `/jobs` | 게시글 만들기 요청. 작업 번호를 돌려줍니다 |
| `GET` | `/jobs/{job_id}` | 작업 상태와 결과 |
| `GET` | `/jobs/{job_id}/files/{name}` | 결과 이미지 파일 |
| `GET` | `/presets` | 고를 수 있는 카메라 각도 목록 |
| `GET` | `/usage-suggestions` | 품목명으로 활용 예시 후보 추천 |

서버는 VM 안 `127.0.0.1:8003`에서 돕니다 (서빙 담당 포트). 개발 중에는 각자 PC에서 SSH 터널로 `localhost:8003`에 연결합니다. 화면 개발 서버(`localhost:8504`)에서 부를 수 있게 CORS를 열어 둡니다.

값의 이름은 모두 `snake_case`입니다.

## POST /jobs

`multipart/form-data`로 보냅니다. 사진 파일이 들어가기 때문입니다.

| 이름 | 형식 | 필수 | 설명 |
| --- | --- | --- | --- |
| `photo` | 파일 | 예 | 제품 사진. jpg, png, webp. 10MB 이하 |
| `product_name` | 문자열 | 예 | 품목. 예: 참도미 |
| `unit` | 문자열 | 예 | 판매 단위. 예: 1마리, kg |
| `price_krw` | 정수 | 예 | 가격(원). 0보다 큰 정수 |
| `usage_example` | 문자열 | 예 | 활용 예시. 예: 회, 구이 |
| `notes` | 문자열 | 아니오 | 기타 |
| `angles` | 문자열, 여러 번 | 아니오 | 고른 각도 id. 고른 만큼 같은 이름으로 여러 번 보냅니다. 없으면 원본 각도 그대로 |

문구 입력 5개(`product_name`부터 `notes`까지)는 문구 생성 함수 `generate_product_captions`의 입력과 같습니다.

응답 (`202`):

```json
{ "job_id": "20261124-153012-a1b2c3" }
```

입력이 잘못되면 `422`와 이유를 돌려주고, 작업은 만들지 않습니다. `detail`은 FastAPI 기본 형식(영어 목록)이 아니라 **화면에 그대로 보여줄 한국어 문장 하나**입니다. 빠진 값, 형식이 틀린 값, 너무 큰 사진 모두 같은 모양입니다.

```json
{ "detail": "가격은 0보다 큰 숫자로 입력해 주세요." }
```

## GET /jobs/{job_id}

아래는 각도 2개를 골라 하나는 끝나고 하나는 실패한 뒤 `ready`가 된 예입니다.

```json
{
  "job_id": "20261124-153012-a1b2c3",
  "status": "ready",
  "message": "",
  "queue_position": null,
  "product_photo": {
    "image_url": "/jobs/20261124-153012-a1b2c3/files/product_photo.png",
    "angle_images": [
      { "angle": "low_angle", "image_url": "/jobs/20261124-153012-a1b2c3/files/angle_low_angle.png", "error": null },
      { "angle": "wide_angle", "image_url": null, "error": "이 각도는 만들지 못했습니다." }
    ],
    "caption": "상품 소개 문구",
    "hashtags": ["#참도미", "#도미회"],
    "error": null
  },
  "usage_example": {
    "image_url": null,
    "caption": "활용 예시 문구",
    "hashtags": ["#도미구이"],
    "error": null
  }
}
```

| 이름 | 설명 |
| --- | --- |
| `status` | `queued`(순서 기다림), `running`(만드는 중), `ready`(완료), `failed`(실패) |
| `message` | `failed`일 때 화면에 보여줄 이유. 그 외에는 빈 문자열 |
| `queue_position` | `queued`일 때 내 앞에 남은 작업 수. 그 외에는 `null` |
| `product_photo` | `running`부터 들어 있습니다. `queued`일 때는 `null` |
| `usage_example` | `ready`일 때만 들어 있고, 그 전에는 `null` |
| `product_photo.image_url` | 대표 사진. 각도를 바꾸지 않은 **원본(보정) 사진**입니다. 화면의 캐러셀 첫 장입니다 |
| `angle_images` | 고른 각도 사진. **고른 순서대로** 들어 있고, `running` 중에는 끝난 각도부터 하나씩 늘어납니다. 각도를 고르지 않았으면 빈 목록 |
| `angle_images[].error` | 그 각도 하나만 실패했을 때의 이유. 이때 그 각도의 `image_url`은 `null`이고, 다른 각도와 게시글은 그대로 이어서 만듭니다 |
| `image_url` | 이미지 주소. 서버 주소 뒤에 붙여 씁니다. 아직 만들지 않는 사진은 `null` |
| `caption` | 문구. `running` 중에는 빈 문자열 |
| `hashtags` | 해시태그 목록. 해시태그 함수가 실패하면 빈 목록이고, 화면은 이때 해시태그 줄을 숨깁니다. 빈 목록이어도 `error`는 `null`입니다 |
| `error` | 이 게시글 하나만 실패했을 때의 이유. 성공이면 `null`. 문구를 만들지 못하면 두 게시글 모두 `"문구를 만들지 못했습니다. 다시 시도해 주세요."`이고, `caption`은 빈 문자열, `hashtags`는 빈 목록입니다. 이때도 사진은 그대로 들어 있습니다 |

없는 작업 번호면 `404`입니다.

## GET /jobs/{job_id}/files/{name}

결과 PNG 파일을 그대로 돌려줍니다. `image_url`을 `<img src>`에 넣으면 됩니다.

## GET /presets

```json
[
  { "id": "wide_angle", "label": "광각", "method": "rotate" },
  { "id": "low_angle", "label": "로우 앵글 (올려다보기)", "method": "orbit" }
]
```

`models/generative_ai/camera_angle_presets/angle_presets.toml`에서 읽습니다. 검증용 `original`은 빼고 13개를 돌려줍니다. 프리셋이 늘어나면 화면도 이 목록을 쓰면 따라갈 수 있습니다.

## GET /usage-suggestions

품목명으로 활용 예시 후보를 최대 3개 돌려줍니다. 화면은 이 목록에서 고르거나 직접 입력하게 합니다. 고르거나 직접 입력한 값은 `POST /jobs`의 `usage_example`로 보냅니다.

| 이름 | 형식 | 필수 | 설명 |
| --- | --- | --- | --- |
| `product_name` | 문자열 (쿼리) | 예 | 품목. 예: 참도미 |

```json
{ "suggestions": ["회", "소금구이", "맑은탕"] }
```

- 품목이 비어 있거나 공백뿐이면 `422`와 `{ "detail": "품목 이름을 입력해 주세요." }`입니다.
- 추천이 실패하면 `200`에 빈 목록(`{ "suggestions": [] }`)입니다. 화면은 직접 입력만 보여 줍니다.
- 그래픽 카드를 쓰지 않고 문구 생성 모델(`gpt-5-mini`)을 씁니다. 그래서 작업 번호 없이 바로 답합니다.

## 처리 방식

- 그래픽 카드 작업은 한 번에 하나씩 합니다. 겹치면 `queued`로 기다립니다.
- 이미지 생성 모델은 서버를 켤 때 한 번만 불러 둡니다. 첫 요청은 모델을 불러오느라 더 걸립니다.
- 각도 한 장에 약 25~70초가 걸립니다 (PR #8 측정). 고른 각도 수만큼 늘어나서, 끝난 각도부터 `angle_images`에 넣습니다.
- 순서는 원본(보정) 사진 → 고른 순서대로 각도 사진 → 문구 → 해시태그입니다.
- 해시태그는 서버에서 만듭니다. 해시태그 API 키가 브라우저에 드러나지 않게 하기 위해서입니다.
- 문구는 `generate_product_captions`가 두 편을 한 번에 만들고, `product_caption`은 실물 소개, `usage_caption`은 활용 예시 게시글에 들어갑니다. 해시태그는 `generate_hashtags`를 한 번 불러 두 게시글에 나눠 넣습니다.
- 긴 변이 1024픽셀보다 큰 사진은 줄인 뒤 처리합니다. 원본 크기 그대로면 VM 메모리를 넘을 수 있습니다.
- 결과는 작업마다 `/data/outputs/jobs/{job_id}/`에 따로 저장해서 서로 덮어쓰지 않습니다.
- 작업 상태는 서버 메모리에 있어서, 서버를 다시 켜면 이전 작업 번호는 `404`가 됩니다.

## 지금 단계에서 비어 있는 값

| 값 | 지금 | 채워지는 때 |
| --- | --- | --- |
| `usage_example.image_url` | `null` | 활용 예시 사진 담당이 정해진 뒤 |
| 사진 보정 | 하지 않음. 대표 사진은 1024로 줄인 원본 | 보정(PR #9)이 합쳐진 뒤 |

## 정해진 것

화면 담당(PR #11 리뷰)과 맞춘 내용입니다.

1. **각도 여러 개:** 실물 소개 게시글 하나에 여러 장을 넣고 옆으로 넘겨 보는 방식(인스타그램 캐러셀)입니다. 첫 장은 원본(보정) 사진, 그 뒤는 고른 각도 사진이고, 어색한 사진은 화면에서 체크를 꺼서 뺍니다.
2. **해시태그:** 화면 담당이 해시태그 함수를 만들고, 서버가 문구를 만든 뒤 불러서 `hashtags`에 넣습니다.
3. **사진 보내는 방법:** `postApi.js`가 사진 파일(`photo`)을 보냅니다.
4. **값 이름:** `postApi.js` 안에서 이 문서 이름(`job_id`, `product_name` 등)으로 옮겨 적습니다. 문서 이름은 그대로입니다.

## 나중에 할 것

- **실패한 쪽만 다시 만들기:** 화면의 "사진 다시 만들기" 버튼용 주소입니다. 첫 서버(AD-27)에는 넣지 않고, 별도 티켓으로 주소와 값을 이 문서에 먼저 더한 뒤 만듭니다.
