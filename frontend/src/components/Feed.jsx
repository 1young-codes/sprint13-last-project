import { useEffect, useRef, useState } from 'react'

// 가게 프로필과 게시글 격자. 게시 뒤 화면과 "내 피드" 탭이 같이 쓴다.
// post 모양은 api/feedApi.js 참고.

const KIND_LABELS = { product: '제품 소개', usage: '활용 예시' }

// 프로필 위쪽: 로고, 가게 이름, 소개글(없으면 숨김), 게시물 수
export function ProfileHead({ storeName, bio, count }) {
  return (
    <header className="profile-head">
      <span className="avatar large">로고</span>
      <div>
        <strong>{storeName}</strong>
        {bio && <p className="profile-bio">{bio}</p>}
        {count !== undefined && <p className="profile-count">게시물 {count}</p>}
      </div>
    </header>
  )
}

// 게시글 격자 (최신이 앞). newCount: 앞에서 몇 개를 "새 게시물"로 표시할지, onOpen(post): 누르면 크게 보기
export function FeedGrid({ posts, newCount = 0, onOpen, tail }) {
  return (
    <div className="grid">
      {posts.map((post, index) => (
        <FeedTile key={post.id} post={post} isNew={index < newCount} onOpen={onOpen} />
      ))}
      {tail}
    </div>
  )
}

function FeedTile({ post, isNew, onOpen }) {
  const label = KIND_LABELS[post.kind] ?? '게시글'
  const content = (
    <>
      {isNew && <span className="tile-badge">새 게시물</span>}
      {post.images.length > 1 && <span className="tile-many" title={`사진 ${post.images.length}장`}>⧉ {post.images.length}</span>}
      {post.images[0] ? <img src={post.images[0]} alt={`${post.productName} ${label}`} /> : <span>[{label} 사진]</span>}
    </>
  )
  if (!onOpen) return <div className="tile">{content}</div>
  return (
    <button type="button" className="tile tile-button" onClick={() => onOpen(post)} aria-label={`${post.productName} ${label} 크게 보기`}>
      {content}
    </button>
  )
}

// 게시글 하나 크게 보기: 왼쪽 사진(여러 장이면 넘겨 보기), 오른쪽 문구. 휴대폰에서는 위아래로 놓인다.
export function FeedPostView({ post, storeName, onClose }) {
  const [index, setIndex] = useState(0)
  const ref = useRef(null)
  const count = post.images.length

  // 피드 아래쪽 게시글을 눌러도 크게 보기가 보이도록 화면을 옮긴다.
  useEffect(() => {
    ref.current?.scrollIntoView?.({ behavior: 'smooth', block: 'start' })
  }, [])
  const current = Math.min(index, Math.max(count - 1, 0))
  const label = KIND_LABELS[post.kind] ?? '게시글'

  return (
    <article className="post feed-post" ref={ref}>
      <div className="post-photo">
        {count > 0 ? <img key={current} src={post.images[current]} alt={`${post.productName} ${label}`} /> : <span>[{label} 사진]</span>}
        {count > 1 && (
          <>
            <button type="button" className="slide-nav prev" aria-label="이전 사진" disabled={current === 0} onClick={() => setIndex(current - 1)}>
              ‹
            </button>
            <button type="button" className="slide-nav next" aria-label="다음 사진" disabled={current === count - 1} onClick={() => setIndex(current + 1)}>
              ›
            </button>
            <span className="slide-count">{current + 1} / {count}</span>
          </>
        )}
      </div>
      <div className="post-body">
        <header className="post-head">
          <span className="avatar">로고</span>
          <strong>{storeName}</strong>
          <span className={`badge ${post.kind}`}>{label}</span>
        </header>
        <div className="post-text">
          {post.caption ? <p>{post.caption}</p> : <p className="hint">문구가 없습니다.</p>}
          {post.hashtags.length > 0 && <p className="tags">{post.hashtags.join(' ')}</p>}
          <p className="hint">{new Date(post.createdAt).toLocaleString('ko-KR', { dateStyle: 'long', timeStyle: 'short' })}</p>
        </div>
        <div className="post-actions">
          <button type="button" className="secondary" onClick={onClose}>닫기</button>
        </div>
      </div>
    </article>
  )
}
