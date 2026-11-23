# 카메라 앵글 바꾸기

사진 한 장과 프리셋 이름을 넣으면, 카메라 앵글을 바꾼 사진을 만듭니다.
ComfyUI 없이 파이썬만으로 돌아가며, ComfyUI-Panorama-Stickers의 방식을 옮겨 왔습니다.

## 두 가지 방식

| 방식 | 하는 일 | 프리셋 |
| --- | --- | --- |
| 회전 (`rotate`) | 원본 사진을 원하는 방향과 렌즈로 다시 찍고, 원본 밖으로 드러난 빈 곳만 생성합니다. 원본이 보이는 곳은 원본 픽셀 그대로입니다. | `wide_angle`, `ultra_wide`, `telephoto`, `dutch_angle`, `tilt_up`, `tilt_down`, `pan_left`, `pan_right`, `fisheye` |
| 3D 이동 (`orbit`) | 깊이를 추정해 카메라를 피사체 중심으로 돌린 뒤, 드러난 빈 곳을 원본을 참고해 채웁니다. | `low_angle`, `high_angle`, `three_quarter_left`, `three_quarter_right` |

회전 방식은 카메라가 제자리에서 돌기만 하므로 피사체의 옆모습은 나오지 않고, 원본과 겹치지 않는 방향(뒤돌아보기)도 만들지 않습니다. 옆모습이나 올려다본 모습이 필요하면 3D 이동 프리셋을 씁니다.

회전 각도는 원본 화각에 대한 비율(`yaw_ratio`, `pitch_ratio`)로 정합니다. 그래서 화각이 좁은 사진에서도 피사체가 화면 밖으로 나가지 않습니다. 원본이 결과 화면의 4분의 1보다 작게 남는 넓은 화각은 한 번에 채우지 않고, 정면 화각을 몇 번에 나눠 넓혀 가며 채웁니다. 빈 곳이 너무 크면 생성 모델이 배경을 투명하게 그려 버리기 때문입니다. 어안은 일반 렌즈로 먼저 채운 뒤 휘게 만듭니다.

## 가상 머신 준비

팀 가상 머신은 용량을 아끼려고 가상환경과 모델 캐시를 공용으로 씁니다. 공용 가상환경은 `/data/venv`, 허깅페이스 캐시는 `~/.bashrc`의 `HF_HOME`(`/data/models/hf_cache`)입니다. 프로젝트 루트에서 실행합니다.

```bash
source /data/venv/bin/activate
pip install -r models/camera_angle/requirements.txt
python -m models.camera_angle.download_models
```

모델 약 16GB는 `HF_HOME` 캐시에 저장되어 팀원과 같이 씁니다. 이미 받아 둔 모델은 다시 받지 않습니다. 결과와 추론 코드 위치는 `.env`의 `CAMERA_ANGLE_OUTPUT_DIR`, `CAMERA_ANGLE_CODE_DIR`로 바꿉니다. 가상 머신에서는 결과를 공용 폴더 `/data/outputs` 아래에 둡니다.

## 실행

```bash
python -m models.camera_angle --list-presets
python -m models.camera_angle --input 사진.jpg --preset dutch_angle --preset low_angle
python -m models.camera_angle --input 사진.jpg --preset all --scene "a fishing boat on a calm sea"
```

결과는 `outputs/camera_angle/<사진 이름>/`에 저장됩니다.

| 옵션 | 뜻 |
| --- | --- |
| `--scene` | 빈 곳을 채울 때 쓸 장면 설명 (영어 권장). 흰 배경처럼 단색 배경 사진은 `"plain white studio background"`처럼 배경을 적어야 생성 모델이 새 풍경을 지어내지 않습니다 |
| `--photo-fov` | 원본 사진의 가로 화각. 주면 깊이 추정을 건너뜁니다 |
| `--skip-generation` | 생성 모델 없이 계산만 확인합니다. 그래픽 카드 없이 회전 프리셋을 볼 때 씁니다 |
| `--save-steps` | 채우기 전 이미지, 빈 곳 마스크 같은 중간 이미지도 저장합니다 |

## 설정 파일

| 파일 | 내용 |
| --- | --- |
| `angle_presets.toml` | 프리셋별 각도, 화각, 렌즈, 카메라 이동량 |
| `settings.toml` | 경로, 결과 크기, 프롬프트, 시드, 단계 수, 깊이 모델 |
| `model_sources.toml` | 내려받을 모델의 저장소, 고정 커밋, 해시 |

새 프리셋은 `angle_presets.toml`에 표 하나를 추가하면 됩니다. 코드는 고칠 필요가 없습니다.

## 파일별 책임

| 파일 | 하는 일 |
| --- | --- |
| `change_camera_angle.py` | 전체 순서 지휘 |
| `camera_math.py` | 카메라 방향과 픽셀 좌표 계산 |
| `rotate_camera.py` | 원본을 새 방향과 렌즈로 다시 찍고 빈 곳 마스크 만들기 |
| `fill_empty_area.py` | 다시 찍은 사진의 빈 곳 채우기 |
| `estimate_depth.py` | 깊이와 원본 화각 추정 |
| `move_camera_in_3d.py` | 카메라를 피사체 중심으로 돌려 새 시점 그리기 |
| `complete_new_viewpoint.py` | 새 시점의 빈 곳을 원본을 참고해 채우기 |
| `qwen_image_model.py` | 이미지 생성 모델 불러오기와 실행 |
| `speed_up_generation.py` | MagCache로 변화가 작은 생성 단계 건너뛰기 |
| `calibrate_mag_cache.py` | MagCache가 쓰는 단계별 크기 비율 재기 |
| `int8_linear_layers.py` | 정수 8비트 가중치 층 |
| `download_models.py` | 모델 내려받기 |

## 생성 속도

`settings.toml`의 `[speed]`에서 두 가지 최적화를 켜고 끕니다.

| 설정 | 하는 일 | 결과 |
| --- | --- | --- |
| `reuse_condition_cache` | 글자와 참고 이미지 부분을 첫 단계에서 계산해 두고 재사용 | 같음 |
| `mag_cache` | 변화가 작은 단계는 계산을 건너뛰고 앞 단계의 변화량을 재사용 | 아주 조금 달라짐 |

MagCache는 모델과 LoRA마다 잰 단계별 크기 비율(`mag_cache_ratios.json`)을 씁니다. 모델, LoRA, 단계 수를 바꾸면 성격이 다른 사진 여러 장으로 다시 잽니다.

```bash
python -m models.camera_angle.calibrate_mag_cache --input 사진1.jpg --input 사진2.jpg --preset all
```

## 검증

그래픽 카드 없이 계산만 확인합니다.

```bash
python -m pytest tests/test_camera_angle.py
```

## 출처

계산 코드는 ComfyUI-Panorama-Stickers(MIT)를 ai-generation 프로젝트에서 옮긴 것입니다. 고지문은 `LICENSE-Panorama-Stickers.txt`에 있습니다.
