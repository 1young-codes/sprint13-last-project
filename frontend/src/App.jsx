import { useState } from 'react'
import Start from './pages/Start.jsx'
import AccountGuide from './pages/AccountGuide.jsx'
import StoreSetup from './pages/StoreSetup.jsx'
import CreatePost from './pages/CreatePost.jsx'
import MyFeed from './pages/MyFeed.jsx'
import { connectInstagram, disconnectInstagram, loadAccount } from './api/authApi.js'

// 계정을 연결한 뒤 보이는 위쪽 탭 3개. 순서대로 보여준다.
const TABS = [
  { id: 'setup', label: '가게 설정', Page: StoreSetup },
  { id: 'post', label: '게시글 만들기', Page: CreatePost },
  { id: 'feed', label: '내 피드', Page: MyFeed },
]

export default function App() {
  const [account, setAccount] = useState(loadAccount) // 연결된 인스타그램 계정 (없으면 null)
  const [showGuide, setShowGuide] = useState(false) // 계정 만드는 방법 화면을 보고 있는지
  const [connecting, setConnecting] = useState(false)
  const [current, setCurrent] = useState('post')

  async function connect() {
    setConnecting(true)
    const connected = await connectInstagram()
    setConnecting(false)
    setAccount(connected)
    setShowGuide(false)
    setCurrent('post')
  }

  function disconnect() {
    disconnectInstagram()
    setAccount(null)
  }

  // 계정을 연결하기 전: 시작하기 화면 또는 계정 만드는 방법 화면
  if (!account) {
    return (
      <div className="app">
        <header className="topbar">
          <span className="brand">우리 가게 SNS</span>
        </header>
        <main className="content">
          {showGuide ? (
            <AccountGuide onBack={() => setShowGuide(false)} />
          ) : (
            <Start onConnect={connect} onShowGuide={() => setShowGuide(true)} connecting={connecting} />
          )}
        </main>
      </div>
    )
  }

  // 계정을 연결한 뒤: 위쪽 탭과 탭별 화면
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
        <span className="account">
          @{account.username}
          {account.isSample && <span className="sample-tag">예시</span>}
          <button type="button" className="link" onClick={disconnect}>
            연결 끊기
          </button>
        </span>
      </header>
      <main className="content">
        <Page />
      </main>
    </div>
  )
}
