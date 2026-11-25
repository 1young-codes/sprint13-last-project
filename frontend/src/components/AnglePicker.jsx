import { useState } from 'react'
import { ANGLE_PRESETS, SECONDS_PER_ANGLE } from '../anglePresets.js'

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
