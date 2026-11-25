// 시작하기: 인스타그램 비즈니스 계정이 있으면 연결하고, 없으면 만드는 방법을 안내한다.
export default function Start({ onConnect, onShowGuide, connecting }) {
  return (
    <section className="page narrow">
      <p className="eyebrow">시작하기</p>
      <h1 className="big">인스타그램 비즈니스 계정이 필요합니다</h1>
      <p className="lead large">
        이 서비스는 만든 게시글을 사장님의 인스타그램 비즈니스 계정에 올립니다. 계정을 연결하면 바로 시작할 수 있습니다.
      </p>

      <div className="card choice">
        <h2>비즈니스 계정이 있어요</h2>
        <button type="button" className="primary wide" onClick={onConnect} disabled={connecting}>
          {connecting ? '연결하는 중…' : '인스타그램 계정 연결하기'}
        </button>
        <p className="note">
          <span aria-hidden="true">🔒</span>
          인스타그램의 공식 로그인 화면으로 이동합니다. 이 서비스는 아이디와 비밀번호를 받거나 저장하지 않습니다.
        </p>
      </div>

      <div className="card choice">
        <h2>아직 비즈니스 계정이 없어요</h2>
        <p>계정 만들기부터 비즈니스 계정으로 바꾸기까지, 화면 사진과 함께 순서대로 안내합니다.</p>
        <button type="button" className="outline" onClick={onShowGuide}>
          계정 만드는 방법 보기
        </button>
        <p className="hint">계정을 만든 뒤 이 화면으로 돌아와 연결하면 됩니다.</p>
      </div>
    </section>
  )
}
