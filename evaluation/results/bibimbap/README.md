# 비빔밥 업스케일·보정 상품 보존 평가 (AD-34)

민기님 활용 예시 결과 `/data/outputs/usage_examples/lora-compare/finish/food_korean_bibimbap`을 평가했습니다.
각도 15개마다 두 쌍을 비교했습니다.

- 업스케일: `1_angle/<각도>.png` → `2_upscale/<각도>.png` (결과를 원본 크기로 줄여 비교)
- 보정: `2_upscale/<각도>.png` → `3_retouch/<각도>/retouched.png`

지표는 `evaluation/product_fidelity.py`가 냅니다.

- `iou`: 전후 상품 영역(BiRefNet)이 겹치는 정도, 1이면 같음
- `structure`: 상품 영역의 무늬·윤곽 유사도, 1이면 같음
- `delta_e`: 상품 영역 평균 색 차이, 0이면 같음 (음식 업종 보정 한도 10)

## 결과

| 단계 | iou 중앙값 | iou 최저 | structure 중앙값 | structure 최저 | delta_e 중앙값 | delta_e 최고 |
| --- | --- | --- | --- | --- | --- | --- |
| 업스케일 | 0.997 | 0.838 (telephoto) | 0.990 | 0.956 (telephoto) | 2.68 | 3.66 (pan_right) |
| 보정 | 1.000 | 0.969 (telephoto) | 1.000 | 1.000 | 0.34 | 2.06 (telephoto) |

사진별 값은 `upscale.csv`, `retouch.csv`에 있습니다.

- 보정은 상품을 거의 바꾸지 않았습니다. 색 차이 최고값도 음식 한도 10보다 훨씬 낮습니다.
- 보정 색 차이는 민기님 보정 기능의 자체 검사값(`report.json`)과 비슷합니다. 확인한 10개 각도에서 차이가 0.1 이하였습니다.
- 업스케일은 보정보다 색 차이가 큽니다(중앙값 2.7). 한도 안이지만 세 지표 중 가장 크게 움직였습니다.
- 업스케일 iou가 0.95보다 낮은 각도는 telephoto 0.838, dutch_angle 0.903, pan_left 0.916입니다.
  telephoto는 사진으로 확인했습니다. 음식은 전후가 같고, 업스케일 전 사진에서만 영역 나누기가 그릇 왼쪽 위 테두리까지 상품으로 잡았습니다.
  즉 낮은 iou는 상품이 바뀐 것이 아니라 영역 나누기가 흔들린 것입니다. 나머지 두 각도는 사진으로 보지 않았습니다.
  그래서 iou 하나만으로 실패를 판정하면 안 되고, 사람이 볼 사진을 고르는 용도로 쓰는 것이 맞습니다.

## 한계

- 사진 한 장(비빔밥)의 결과이고, 원본도 생성 이미지입니다. 실제 사장님 사진의 보정 결과는 아직 없습니다.
- 각도 변환은 구도가 바뀌어 이 지표로 비교할 수 없어 뺐습니다.
- 그래픽 카드에서 한 단계에 약 35초(모델 불러오기 포함) 걸렸습니다.

## 다시 돌리기 (VM)

```bash
D=/data/outputs/usage_examples/lora-compare/finish/food_korean_bibimbap
for f in $D/2_upscale/*.png; do a=$(basename $f .png)
  echo "$D/1_angle/$a.png,$f" >> upscale_pairs.csv
  echo "$f,$D/3_retouch/$a/retouched.png" >> retouch_pairs.csv
done
HF_HOME=/data/models/hf_cache /data/venvs/photo_retouch/bin/python -m evaluation.product_fidelity upscale_pairs.csv --out upscale.csv
HF_HOME=/data/models/hf_cache /data/venvs/photo_retouch/bin/python -m evaluation.product_fidelity retouch_pairs.csv --out retouch.csv
```
