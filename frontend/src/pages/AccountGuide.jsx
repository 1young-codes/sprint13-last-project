// 계정 만드는 방법: 인스타그램 가입부터 비즈니스 계정으로 바꾸기까지 단계별 안내.
// 메뉴 이름은 인스타그램 앱 버전에 따라 조금 다를 수 있다. 화면 사진은 실제 캡처로 채울 자리다.
const STEPS = [
  {
    title: '인스타그램 앱을 설치하고 가입하기',
    body: '휴대폰에서 인스타그램 앱을 설치하고, 전화번호나 이메일로 새 계정을 만듭니다. 사용자 이름은 가게 이름으로 정하면 손님이 찾기 쉽습니다.',
    shot: '가입 화면',
  },
  {
    title: '프로필 사진과 소개 넣기',
    body: '프로필 편집에서 가게 간판이나 대표 상품 사진을 프로필 사진으로 넣고, 소개에 영업시간과 위치를 적습니다. 밴드 주소도 여기에 넣을 수 있습니다.',
    shot: '프로필 편집 화면',
  },
  {
    title: '프로페셔널 계정으로 바꾸기',
    body: '프로필 화면 오른쪽 위의 메뉴(≡)를 누르고 설정에서 "계정 유형 및 도구" → "프로페셔널 계정으로 전환"을 고릅니다.',
    shot: '설정 메뉴 화면',
  },
  {
    title: '"비즈니스"를 고르고 업종 정하기',
    body: '크리에이터와 비즈니스 중 "비즈니스"를 고르고, 가게에 맞는 카테고리(예: 수산물 가게, 베이커리)를 고릅니다. 연락처는 필요한 것만 공개해도 됩니다.',
    shot: '비즈니스 선택 화면',
  },
]

export default function AccountGuide({ onBack }) {
  return (
    <section className="page narrow">
      <button type="button" className="link back" onClick={onBack}>
        ← 처음 화면으로
      </button>
      <p className="eyebrow">계정 만드는 방법</p>
      <h1 className="big">인스타그램 비즈니스 계정 만들기</h1>
      <p className="lead large">휴대폰 하나로 10분 정도면 끝납니다. 아래 순서대로 따라 해 주세요.</p>

      <ol className="guide">
        {STEPS.map((step, index) => (
          <li key={step.title} className="card guide-step">
            <span className="guide-num">{index + 1}</span>
            <div className="guide-text">
              <h2>{step.title}</h2>
              <p>{step.body}</p>
            </div>
            <div className="guide-shot" aria-label={`${step.shot} 사진 자리`}>
              [{step.shot}]
            </div>
          </li>
        ))}
      </ol>

      <div className="card choice done-guide">
        <h2>계정을 다 만들었나요?</h2>
        <p>처음 화면으로 돌아가 "인스타그램 계정 연결하기"를 누르면 됩니다.</p>
        <button type="button" className="primary wide" onClick={onBack}>
          연결하러 가기
        </button>
      </div>
    </section>
  )
}
