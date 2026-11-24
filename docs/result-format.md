# 결과 파일 형식

화면이 읽는 결과물의 위치와 이름입니다. 이 형식대로 저장하면 화면에 자동으로 보입니다.

## 저장 위치

`/data/outputs/demo/`

## 파일

| 파일 | 내용 | 화면에 쓰나 |
|---|---|---|
| `product_photo.png` | 실물 소개 컷 (원본 제품을 보존) | 쓴다 |
| `usage_example.png` | 활용 컷 (연출) | 쓴다 |
| `vm_smoke.png` | 모델 실행 확인용 | 쓰지 않는다 |
| `result.json` | 상태와 문구 | 쓴다 |

이미지는 정사각형, 한 변 512픽셀 이상입니다.

## result.json

```json
{
  "status": "ready",
  "message": "",
  "product_name": "도미",
  "description": "한 줄 설명",
  "product_caption": "상품 소개 문구",
  "usage_caption": "활용 예시 문구",
  "model_id": "사용한 모델 이름",
  "seed": 0
}
```

`status` 값은 세 가지입니다.

- `running`: 만드는 중. 화면은 "만드는 중"을 보여줍니다.
- `ready`: 완료. 화면은 이미지 두 장과 문구를 보여줍니다.
- `failed`: 실패. 화면은 `message`의 내용을 보여줍니다.

`seed`를 모르면 `null`로 둡니다.

`result.json`이 없으면 화면은 샘플 이미지와 샘플 문구를 보여줍니다.
