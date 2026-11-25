// 인스타그램 계정 연결.
// 실제 연결(인스타그램 공식 로그인 화면으로 이동)은 팀 결정 전이라 지금은 연결된 척만 한다.
// 실제 연결을 붙이면 이 파일만 바꾼다. 이 서비스는 아이디와 비밀번호를 받거나 저장하지 않는다.

const STORAGE_KEY = 'store-sns.account'

export function loadAccount() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved ? JSON.parse(saved) : null
  } catch {
    return null
  }
}

export async function connectInstagram() {
  const account = { username: 'our_store_sample', isSample: true }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(account))
  } catch {
    // 저장이 안 되는 브라우저(사생활 보호 모드 등)에서도 이번 방문 동안은 쓸 수 있다.
  }
  return account
}

export function disconnectInstagram() {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    // 지울 것이 없으면 그대로 둔다.
  }
}
