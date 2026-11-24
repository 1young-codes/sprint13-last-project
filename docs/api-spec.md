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

입력이 잘못되면 `422`와 이유를 돌려주고, 작업은 만들지 않습니다.

```json
{ "detail": "price_krw는 0보다 큰 정수여야 합니다." }
```

## GET /jobs/{job_id}

```json
{
  "job_id": "20261124-153012-a1b2c3",
  "status": "ready",
  "message": "",
  "queue_position": null,
  "product_photo": {
    "image_url": "/jobs/20261124-153012-a1b2c3/files/product_photo.png",
    "angle_images": [
      { "angle": "low_angle", "image_url": "/jobs/20261124-153012-a1b2c3/files/angle_low_angle.png" }
    ],
    "caption": "상품 소개 문구",
    "hashtags": [],
    "error": null
  },
  "usage_example": {
    "image_url": null,
    "caption": "활용 예시 문구",
    "hashtags": [],
    "error": null
  }
}
```

| 이름 | 설명 |
| --- | --- |
| `status` | `queued`(순서 기다림), `running`(만드는 중), `ready`(완료), `failed`(실패) |
| `message` | `failed`일 때 화면에 보여줄 이유. 그 외에는 빈 문자열 |
| `queue_position` | `queued`일 때 내 앞에 남은 작업 수. 그 외에는 `null` |
| `product_photo`, `usage_example` | `ready`일 때만 들어 있고, 그 전에는 `null` |
| `image_url` | 이미지 주소. 서버 주소 뒤에 붙여 씁니다. 아직 만들지 않는 사진은 `null` |
| `angle_images` | 고른 각도마다 한 장씩. 각도를 고르지 않았으면 빈 목록 |
| `caption` | 문구 |
| `hashtags` | 해시태그 목록 |
| `error` | 이 게시글 하나만 실패했을 때의 이유. 성공이면 `null` |

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

## 처리 방식

- 그래픽 카드 작업은 한 번에 하나씩 합니다. 겹치면 `queued`로 기다립니다.
- 이미지 생성 모델은 서버를 켤 때 한 번만 불러 둡니다. 첫 요청은 모델을 불러오느라 더 걸립니다.
- 각도 한 장에 약 25~70초가 걸립니다 (PR #8 측정). 고른 각도 수만큼 늘어납니다.
- 긴 변이 1024픽셀보다 큰 사진은 줄인 뒤 처리합니다. 원본 크기 그대로면 VM 메모리를 넘을 수 있습니다.
- 결과는 작업마다 `/data/outputs/jobs/{job_id}/`에 따로 저장해서 서로 덮어쓰지 않습니다.
- 작업 상태는 서버 메모리에 있어서, 서버를 다시 켜면 이전 작업 번호는 `404`가 됩니다.

## 지금 단계에서 비어 있는 값

| 값 | 지금 | 채워지는 때 |
| --- | --- | --- |
| `caption` | `[임시 문구]`로 시작하는 문장 | 문구 생성 함수(PR #6, dev에 합쳐짐)를 서버에 연결한 뒤. `product_caption`은 실물 소개, `usage_caption`은 활용 예시 문구로 들어갑니다 |
| `hashtags` | 빈 목록 | 아래 "정할 것" 참고 |
| `usage_example.image_url` | `null` | 활용 예시 사진 담당이 정해진 뒤 |
| 사진 보정 | 하지 않음 | 보정(PR #9)이 합쳐진 뒤 |

## 정할 것

화면 담당과 맞춰야 하는 부분입니다. 정해지면 이 문서를 고칩니다.

1. **각도를 여러 개 고르면 어떻게 보여줄지.** 지금 안은 `product_photo.image_url`에 첫 각도 사진을 넣고, 전체는 `angle_images`에 모두 넣는 것입니다. 화면에서 여러 장을 넘겨 보게 할지, 하나를 골라 게시하게 할지 정해야 합니다.
2. **해시태그는 누가 만들지.** 문구 생성 함수는 지금 `product_caption`, `usage_caption` 두 개만 돌려줍니다.
3. **사진 보내는 방법.** 지금 `postApi.js`는 브라우저 안에서만 쓰는 임시 주소(`photoUrl`)를 넘기는데, 서버는 이 주소를 읽을 수 없습니다. 파일 자체(`photo`)를 보내야 합니다.
4. **이름 바꾸기.** `postApi.js`는 `jobId`, `productName`처럼 쓰고 있어서, 이 문서의 `job_id`, `product_name`으로 바꾸거나 파일 안에서 옮겨 적어야 합니다.
