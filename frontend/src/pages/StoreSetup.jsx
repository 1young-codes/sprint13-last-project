import Stepper from '../components/Stepper.jsx'

// 가게 설정 (처음 한 번): ① 입력 → ② 결과
// 지금은 뼈대만 있다. 입력 칸과 로고 고르기는 다음 작업에서 채운다.
export default function StoreSetup() {
  return (
    <section className="page">
      <Stepper steps={['입력', '결과']} current={0} />
      <h1>가게를 알려주세요</h1>
      <p className="lead">처음 한 번만 입력합니다. 이 정보로 프로필 사진에 쓸 로고와 소개글을 만듭니다.</p>
      <div className="card placeholder">가게 이름, 업종, 분위기, 한 줄 소개 입력 칸이 들어갈 자리</div>
    </section>
  )
}
