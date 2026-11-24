# 업스케일

보정까지 끝난 사진을 크게 키우는 단계입니다. 흐름은 각도 프리셋 → 보정 → 업스케일입니다.
후보 모델 5종을 같은 사진으로 비교해, **실사 원본을 가장 잘 지키는 모델 하나**를 고릅니다.

## 후보 모델

| 폴더 | 모델 | 방식 | 라이선스 | 특이 사항 |
| --- | --- | --- | --- | --- |
| `fidesr/` | FiDeSR (CVPR 2026) | SD2.1 + LoRA, 한 단계 확산 | Apache-2.0 | torch 2.0.1 고정 |
| `seedvr2/` | SeedVR2-3B (ICLR 2026) | 한 단계 비디오 확산 트랜스포머 | Apache-2.0 | 공식 코드는 L4에서 안 돌아 numz 구현 사용 |
| `vosr/` | VOSR 2.0 (CVPR 2026) | 이미지만 보는 1.4B 확산 트랜스포머 | Apache-2.0 | 512 학습, 타일 필수 |
| `pixrestore/` | PixRestore XL | 픽셀 공간 확산 트랜스포머, 한 단계 | Apache-2.0 | 1배 복원 모델 (바이큐빅으로 키운 뒤 복원) |
| `varestorer/` | VARestorer (ICLR 2026) | 시각 자기회귀(Infinity-2B) 한 단계 | MIT | 기본 프롬프트가 고정 문장이라 빈 프롬프트로 바꿈 |

각 모델 폴더는 서로 독립입니다. 폴더마다 `setup.toml`(코드·가중치 커밋, 가중치 연결 위치, 실행 명령)과 `requirements.txt`가 있고, 가상환경도 따로 만듭니다. 모델마다 torch 버전이 2.0.1부터 2.7.1까지 달라 한 환경에 함께 깔 수 없기 때문입니다.

## 준비

가상환경은 `uv`로 만듭니다. 이미 끝난 단계는 건너뛰므로 여러 번 실행해도 됩니다.

```bash
python -m models.generative_ai.upscales.setup_model              # benchmark와 모든 모델
python -m models.generative_ai.upscales.setup_model vosr         # 하나만
```

경로는 `settings.toml`에 프로젝트 루트 기준으로 적혀 있습니다. 가상 머신에서는 `.env`로 공용 폴더를 가리킵니다.

| 환경 변수 | 뜻 | 가상 머신 값 |
| --- | --- | --- |
| `UPSCALE_CODE_DIR` | 모델별 추론 코드 | `/data/models/vendor/upscales` |
| `UPSCALE_VENV_DIR` | 모델별 가상환경 | `/data/venvs/upscales` |
| `UPSCALE_OUTPUT_DIR` | 결과와 보고서 | `/data/outputs/upscales` |
| `UPSCALE_BENCH_PHOTOS_DIR` | 비교용 실사 사진 | `/data/datasets/upscale_bench` |
| `UPSCALE_CAMERA_ANGLE_DIR` | 각도 프리셋 결과 | `/data/outputs/camera_angle` |
| `UPSCALE_METRIC_MODELS_DIR` | 얼굴·글자 인식 모델 | `/data/models/metrics` |

가중치는 허깅페이스 캐시(`HF_HOME`)에 커밋을 고정해 받습니다.

## 실행

모델 하나로 폴더 안 사진을 키웁니다. 걸린 시간과 최대 그래픽 메모리를 `run_info.json`에 남깁니다.

```bash
python -m models.generative_ai.upscales.run_model vosr --input 사진폴더 --scale 2
```

## 비교

`benchmark` 가상환경의 파이썬으로 실행합니다.

```bash
python -m models.generative_ai.upscales.benchmark
python -m models.generative_ai.upscales.benchmark --model vosr --condition x4_real
```

| 조건 | 입력 | 보는 것 |
| --- | --- | --- |
| `x4_real` | 실사 원본 ×4 | 정답이 없으므로 결과를 되돌려 줄여 입력과 비교, 얼굴·글자 보존, 영역별 비교 |
| `x2_pipeline` | 각도 프리셋 결과 ×2 | 실제 서비스 흐름의 속도와 메모리, 되돌려 줄인 비교 |
| `x2_synthetic` | 각도 프리셋 결과를 절반으로 줄였다가 ×2 | 정답과 직접 비교하는 PSNR, SSIM, LPIPS, DISTS |

| 지표 | 뜻 |
| --- | --- |
| `lr_*` | 결과를 입력 크기로 되돌려 줄였을 때 입력과 얼마나 같은가. 없던 것을 지어내면 떨어집니다 |
| `gt_*` | 정답 사진과 얼마나 같은가 |
| `musiq`, `maniqa`, `clipiqa` | 정답 없이 보는 화질 점수 |
| `face_similarity` | 얼굴 인식 모델로 본 동일인 유사도 |
| `ocr_cer` | 글자 인식 결과의 오류율 |
| `detail_gain` | 바이큐빅보다 고주파가 몇 배인가. 크게 높으면 디테일을 지어냈을 수 있습니다 |
| `region_*_lr_psnr` | 얼굴, 머리카락, 비늘, 글자 영역만 따로 본 되돌려 줄인 비교 |

선정 규칙은 `benchmark/benchmark.toml`에 있습니다. 먼저 통과선(원본 일치, 얼굴, 글자)을 하나라도 못 넘으면 탈락합니다. 남은 모델끼리 지표별 순위를 가중 평균해 가장 앞선 모델을 고릅니다. 결과는 `<output_dir>/benchmark/report.md`, 영역 확대 그림은 같은 폴더의 `sheet_*.png`입니다.

## 검증

그래픽 카드 없이 설정과 계산만 확인합니다.

```bash
python -m pytest models/generative_ai/upscales
```

## 라이선스 주의

비교에 쓰는 `pyiqa`는 비상업 라이선스이고, `insightface` 얼굴 모델은 비상업 연구용입니다. 비교에만 쓰고 서비스에는 넣지 않습니다.
