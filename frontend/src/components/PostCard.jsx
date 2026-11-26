import { useState } from 'react'
import { postSlides } from '../postSlides.js'

// 인스타그램 PC 화면처럼 왼쪽 사진, 오른쪽 문구로 보여주는 게시글 카드.
// kind: 'product'(제품 소개) 또는 'usage'(활용 예시)
// result: 서버 응답의 product_photo 또는 usage_example
// 제품 소개는 사진 여러 장을 옆으로 넘겨 보고(캐러셀), 어색한 사진은 체크를 꺼서 뺀다.
//   pendingAngles: 아직 만드는 중인 각도 id 목록, excluded: 뺀 사진 key 목록, onToggle(key)
const LABELS = {
  product: { badge: '제품 소개', empty: '[보정된 제품 사진]' },
  usage: { badge: '활용 예시', empty: '[제품 활용 예시 사진]' },
}

export default function PostCard({ kind, storeName, result, pendingAngles = [], excluded = [], onToggle }) {
  const label = LABELS[kind]
  const slides = postSlides(result, pendingAngles)
  const [index, setIndex] = useState(0)
  const current = Math.min(index, slides.length - 1)
  const slide = slides[current]
  const many = slides.length > 1
  const included = !excluded.includes(slide.key)

  return (
    <article className="post">
      <div className="post-photo">
        <SlideView slide={slide} empty={label.empty} dimmed={many && slide.url && !included} />

        {many && (
          <>
            <button
              type="button"
              className="slide-nav prev"
              aria-label="이전 사진"
              disabled={current === 0}
              onClick={() => setIndex(current - 1)}
            >
              ‹
            </button>
            <button
              type="button"
              className="slide-nav next"
              aria-label="다음 사진"
              disabled={current === slides.length - 1}
              onClick={() => setIndex(current + 1)}
            >
              ›
            </button>
            <span className="slide-count">{current + 1} / {slides.length}</span>
            <div className="slide-dots" aria-hidden="true">
              {slides.map((item, i) => (
                <span key={item.key} className={[i === current && 'on', excluded.includes(item.key) && 'off'].filter(Boolean).join(' ')} />
              ))}
            </div>
          </>
        )}
      </div>

      <div className="post-body">
        <header className="post-head">
          <span className="avatar">로고</span>
          <strong>{storeName}</strong>
          <span className={`badge ${kind}`}>{label.badge}</span>
        </header>

        {many && (
          <div className="slide-check">
            <span className="slide-name">{slide.label}</span>
            {slide.url ? (
              <label>
                <input type="checkbox" checked={included} onChange={() => onToggle?.(slide.key)} />
                이 사진 넣기
              </label>
            ) : (
              <span className="hint">{slide.pending ? '만드는 중' : '넣을 수 없음'}</span>
            )}
          </div>
        )}

        {result.error ? (
          <p className="error">{result.error}</p>
        ) : (
          <div className="post-text">
            {result.caption ? <p>{result.caption}</p> : <p className="hint">문구를 만드는 중입니다…</p>}
            {result.hashtags?.length > 0 && <p className="tags">{result.hashtags.join(' ')}</p>}
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

function SlideView({ slide, empty, dimmed }) {
  if (slide.pending) {
    return (
      <span className="slide-wait">
        <span className="spinner" aria-hidden="true" />
        {slide.label} 사진을 만드는 중입니다
      </span>
    )
  }
  if (slide.error) return <span className="slide-wait">{slide.label}: {slide.error}</span>
  if (!slide.url) return <span>{empty}</span>
  // key: 사진을 넘길 때 새 사진이 받아지기 전까지 이전 사진이 남아 보이지 않게 한다
  return <img key={slide.url} src={slide.url} alt={slide.label} className={dimmed ? 'dimmed' : undefined} />
}
