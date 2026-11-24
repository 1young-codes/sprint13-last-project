import { useState } from 'react'
import StoreSetup from './pages/StoreSetup.jsx'
import CreatePost from './pages/CreatePost.jsx'
import MyFeed from './pages/MyFeed.jsx'

// 화면 시안의 위쪽 탭 3개. 순서대로 보여준다.
const TABS = [
  { id: 'setup', label: '가게 설정', Page: StoreSetup },
  { id: 'post', label: '게시글 만들기', Page: CreatePost },
  { id: 'feed', label: '내 피드', Page: MyFeed },
]

export default function App() {
  const [current, setCurrent] = useState('post')
  const { Page } = TABS.find((tab) => tab.id === current)

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">우리 가게 SNS</span>
        <nav className="tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={tab.id === current ? 'tab active' : 'tab'}
              aria-current={tab.id === current ? 'page' : undefined}
              onClick={() => setCurrent(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="content">
        <Page />
      </main>
    </div>
  )
}
