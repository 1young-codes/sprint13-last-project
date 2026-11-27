// 카메라 각도 프리셋. id는 models/generative_ai/camera_angle_presets/angle_presets.toml, 서버의 GET /presets와 같다.
// (검증용 original은 빼고, 아무것도 고르지 않으면 원본 그대로 간다는 팀 결정을 따른다.)
export const ANGLE_PRESETS = [
  { id: 'wide_angle', label: '광각' },
  { id: 'ultra_wide', label: '초광각' },
  { id: 'telephoto', label: '망원' },
  { id: 'dutch_angle', label: '더치 앵글' },
  { id: 'tilt_up', label: '위로 기울이기' },
  { id: 'tilt_down', label: '아래로 기울이기' },
  { id: 'pan_left', label: '왼쪽으로 돌리기' },
  { id: 'pan_right', label: '오른쪽으로 돌리기' },
  { id: 'fisheye', label: '어안 렌즈' },
  { id: 'low_angle', label: '올려다보기' },
  { id: 'high_angle', label: '내려다보기' },
  { id: 'three_quarter_left', label: '왼쪽 비스듬히' },
  { id: 'three_quarter_right', label: '오른쪽 비스듬히' },
]

export const ANGLE_LABELS = Object.fromEntries(ANGLE_PRESETS.map((preset) => [preset.id, preset.label]))

// 실제 모델 서버 측정(11/27, PR #24, 도미 사진): 각도 없음 18.3초, 각도 1개 105.1초.
// 각도 없이 대표 사진 보정, 문구, 해시태그까지 걸리는 대략적인 시간(초)
export const BASE_SECONDS = 20
// 첫 각도는 오래 걸린다 (105.1초 - 18.3초 = 약 87초)
export const FIRST_ANGLE_SECONDS = 87
// 두 번째 각도부터 한 장에 걸리는 시간(초). 서버 측정(PR #16) 29~44초 중 큰 값. 여러 장은 아직 다시 재지 않았다.
export const SECONDS_PER_ANGLE = 44

// 고른 각도 수로 예상 시간을 "약 N분"으로 알려준다. 짧게 말하는 것보다 길게 말하는 편이 낫다.
export function estimateMinutes(angleCount) {
  const angleSeconds = angleCount > 0 ? FIRST_ANGLE_SECONDS + (angleCount - 1) * SECONDS_PER_ANGLE : 0
  return Math.max(1, Math.ceil((BASE_SECONDS + angleSeconds) / 60))
}
