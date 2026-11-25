import { useState } from 'react'

// 카메라 각도 프리셋. id는 models/generative_ai/camera_angle_presets/angle_presets.toml과 같다.
// (검증용 original은 빼고, 아무것도 고르지 않으면 원본 그대로 간다는 팀 결정을 따른다.)
const ANGLE_PRESETS = [
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

// 각도 하나를 만드는 데 걸리는 대략적인 시간(초). PR #5 측정값 21~38초 중 큰 값.
const SECONDS_PER_ANGLE = 40

// 여러 개를 고를 수 있는 각도 선택 칸. 처음에는 접혀 있다.
// selected: 고른 id 목록, onChange(ids)
export default function AnglePicker({ selected, onChange }) {
  const [open, setOpen] = useState(selected.length > 0)

  function toggle(id) {
    onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id])
  }

  const minutes = Math.ceil((selected.length * SECONDS_PER_ANGLE) / 60)

  return (
    <div className="angle">
      <button type="button" className="angle-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span>{open ? '▾' : '▸'} 각도 바꾸기 (선택)</span>
        <span className="hint">
          {selected.length === 0 ? '고르지 않으면 원본 각도 그대로 만듭니다' : `${selected.length}개 선택`}
        </span>
      </button>
      {open && (
        <>
          <div className="chips">
            {ANGLE_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={selected.includes(preset.id) ? 'chip on' : 'chip'}
                aria-pressed={selected.includes(preset.id)}
                onClick={() => toggle(preset.id)}
              >
                {preset.label}
              </button>
            ))}
          </div>
          <p className="hint">
            {selected.length === 0
              ? '여러 개를 고를 수 있습니다. 고른 각도마다 사진을 한 장씩 더 만듭니다.'
              : `고른 각도가 많을수록 오래 걸립니다. 지금 고른 것은 약 ${minutes}분 더 걸립니다.`}
          </p>
          {selected.length > 0 && (
            <button type="button" className="link" onClick={() => onChange([])}>
              모두 지우기
            </button>
          )}
        </>
      )}
    </div>
  )
}
