// 내 피드: 손님이 가게 프로필에 들어왔을 때 보게 될 모습
// 지금은 뼈대만 있다. 프로필과 게시글 목록은 다음 작업에서 채운다.
export default function MyFeed() {
  return (
    <section className="page">
      <h1>내 피드</h1>
      <p className="lead">손님이 가게 프로필에 들어왔을 때 보게 될 모습입니다.</p>
      <div className="card placeholder">로고, 가게 이름, 소개글과 게시글 목록이 들어갈 자리</div>
    </section>
  )
}
