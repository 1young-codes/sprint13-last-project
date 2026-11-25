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

// 각도 한 장에 걸리는 대략적인 시간(초). 서버 측정(PR #16, 그래픽 카드 위) 29~44초 중 큰 값에 가깝게.
export const SECONDS_PER_ANGLE = 40

// 각도 없이 대표 사진, 문구, 해시태그까지 걸리는 대략적인 시간(초)
export const BASE_SECONDS = 20

// 고른 각도 수로 예상 시간을 "약 N분"으로 알려준다.
export function estimateMinutes(angleCount) {
  return Math.max(1, Math.ceil((BASE_SECONDS + angleCount * SECONDS_PER_ANGLE) / 60))
}
