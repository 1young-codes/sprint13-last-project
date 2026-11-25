// 화면 위쪽의 단계 표시 (예: ① 입력 ─ ② 결과 ─ ③ 게시)
// steps: 단계 이름 목록, current: 지금 단계 번호(0부터)
export default function Stepper({ steps, current }) {
  return (
    <ol className="stepper">
      {steps.map((name, index) => {
        const state = index < current ? 'done' : index === current ? 'now' : 'todo'
        return (
          <li key={name} className={`step ${state}`}>
            <span className="dot">{state === 'done' ? '✓' : index + 1}</span>
            <span className="name">{name}</span>
          </li>
        )
      })}
    </ol>
  )
}
