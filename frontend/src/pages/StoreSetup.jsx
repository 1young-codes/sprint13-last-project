import { useState } from 'react'
import Stepper from '../components/Stepper.jsx'
import { MAX_STORE_NAME } from '../api/storeApi.js'

// 가게 설정 (처음 한 번): ① 입력 → ② 결과
// 지금은 가게 이름만 받는다. 업종, 분위기, 한 줄 소개와 로고 만들기는 다음 작업에서 채운다.
//   store: 저장된 가게 정보, onSave(store): 저장하기
export default function StoreSetup({ store, onSave }) {
  const [name, setName] = useState(store.name)
  const [saved, setSaved] = useState(false)
  const trimmed = name.trim()
  const changed = trimmed !== store.name

  function save(event) {
    event.preventDefault()
    onSave({ ...store, name: trimmed })
    setSaved(true)
  }

  return (
    <section className="page">
      <Stepper steps={['입력', '결과']} current={0} />
      <h1>가게를 알려주세요</h1>
      <p className="lead">처음 한 번만 입력합니다. 이 정보로 프로필 사진에 쓸 로고와 소개글을 만듭니다.</p>

      <form className="card form store-form" onSubmit={save}>
        <label className="field-label" htmlFor="storeName">가게 이름</label>
        <input
          id="storeName"
          type="text"
          placeholder="예: 바다수산"
          maxLength={MAX_STORE_NAME}
          value={name}
          onChange={(event) => {
            setName(event.target.value)
            setSaved(false)
          }}
        />
        <p className="hint">게시글 위쪽과 가게 프로필에 이 이름이 나옵니다.</p>

        <div className="footer">
          <span className="hint" role="status">{saved && !changed ? '저장했습니다.' : ''}</span>
          <button type="submit" className="primary" disabled={!trimmed || !changed}>저장</button>
        </div>
      </form>

      <div className="card placeholder">업종, 분위기, 한 줄 소개 입력 칸이 들어갈 자리</div>
    </section>
  )
}
