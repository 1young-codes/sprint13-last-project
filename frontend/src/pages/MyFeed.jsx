import { useState } from 'react'
import { FeedGrid, FeedPostView, ProfileHead } from '../components/Feed.jsx'

// 내 피드: 손님이 가게 프로필에 들어왔을 때 보게 될 모습. 게시한 게시글이 최신부터 쌓인다.
//   storeName: 가게 이름, storeBio: 한 줄 소개, feed: 게시한 게시글 목록, onClearFeed(): 피드 비우기, onGoTo(tab): 다른 탭으로
export default function MyFeed({ storeName, storeBio, feed = [], onClearFeed, onGoTo }) {
  const [openId, setOpenId] = useState(null)
  const opened = feed.find((post) => post.id === openId)

  function clear() {
    if (window.confirm('피드에 쌓인 게시글을 모두 지울까요? 이 브라우저에서만 지워집니다.')) {
      setOpenId(null)
      onClearFeed?.()
    }
  }

  return (
    <section className="page">
      <h1>내 피드</h1>
      <p className="lead">손님이 가게 프로필에 들어왔을 때 보게 될 모습입니다. 게시글을 누르면 크게 볼 수 있습니다.</p>

      {opened && <FeedPostView key={opened.id} post={opened} storeName={storeName} onClose={() => setOpenId(null)} />}

      <div className="card profile">
        <ProfileHead storeName={storeName} bio={storeBio} count={feed.length} />
        {feed.length > 0 ? (
          <FeedGrid posts={feed} onOpen={(post) => setOpenId(post.id)} />
        ) : (
          <div className="feed-empty">
            <p>아직 게시한 글이 없습니다.</p>
            <button type="button" className="primary" onClick={() => onGoTo?.('post')}>게시글 만들기</button>
          </div>
        )}
      </div>

      {feed.length > 0 && (
        <div className="footer end">
          <button type="button" className="link" onClick={clear}>피드 비우기</button>
        </div>
      )}
      <p className="hint">지금은 이 브라우저에만 저장됩니다. 최근 게시글 30개까지 보입니다.</p>
    </section>
  )
}
