// 인스타그램 PC 화면처럼 왼쪽 사진, 오른쪽 문구로 보여주는 게시글 카드.
// kind: 'product'(제품 소개) 또는 'usage'(활용 예시)
const LABELS = {
  product: { badge: '제품 소개', empty: '[보정된 제품 사진]' },
  usage: { badge: '활용 예시', empty: '[제품 활용 예시 사진]' },
}

export default function PostCard({ kind, storeName, result }) {
  const label = LABELS[kind]

  return (
    <article className="post">
      <div className="post-photo">
        {result.image_url ? (
          <img src={result.image_url} alt={label.badge} />
        ) : (
          <span>{label.empty}</span>
        )}
      </div>
      <div className="post-body">
        <header className="post-head">
          <span className="avatar">로고</span>
          <strong>{storeName}</strong>
          <span className={`badge ${kind}`}>{label.badge}</span>
        </header>
        {result.error ? (
          <p className="error">이 게시글을 만들지 못했습니다. {result.error}</p>
        ) : (
          <div className="post-text">
            <p>{result.caption}</p>
            <p className="tags">{result.hashtags.join(' ')}</p>
          </div>
        )}
        <div className="post-icons" aria-hidden="true">
          <span>♡</span>
          <span>💬</span>
          <span className="right">🔖</span>
        </div>
        <div className="post-actions">
          <button type="button" className="secondary" disabled title="이미지 생성 기능과 연결한 뒤 쓸 수 있습니다">
            사진 다시 만들기
          </button>
          <button type="button" className="secondary" disabled title="문구 생성 기능과 연결한 뒤 쓸 수 있습니다">
            문구 고치기
          </button>
        </div>
      </div>
    </article>
  )
}
