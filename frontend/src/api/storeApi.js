// 가게 정보 (처음 한 번 입력하고 게시글마다 쓴다).
// 서버에 가게 정보를 저장하는 기능이 생기기 전까지는 이 브라우저에만 저장한다.
// 서버에 저장하게 되면 이 파일만 바꾼다. 화면(컴퓨터, 휴대폰)은 그대로 쓴다.

const STORAGE_KEY = 'store-sns.store'
export const MAX_STORE_NAME = 30 // 인스타그램 이름 칸 길이 제한

const EMPTY_STORE = { name: '' }

export function loadStore() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved ? { ...EMPTY_STORE, ...JSON.parse(saved) } : EMPTY_STORE
  } catch {
    return EMPTY_STORE
  }
}

// 앞뒤 빈칸을 지우고 저장한다. 저장한 값을 돌려준다.
export function saveStore(store) {
  const cleaned = { ...store, name: store.name.trim().slice(0, MAX_STORE_NAME) }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cleaned))
  } catch {
    // 저장이 안 되는 브라우저(사생활 보호 모드 등)에서도 이번 방문 동안은 쓸 수 있다.
  }
  return cleaned
}
