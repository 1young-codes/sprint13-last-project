import { useState } from 'react'
import Stepper from '../components/Stepper.jsx'
import { MAX_STORE_BIO, MAX_STORE_NAME } from '../api/storeApi.js'

// 가게 설정 (처음 한 번): ① 입력 → ② 결과
// 지금은 가게 이름과 한 줄 소개를 받는다. 업종, 분위기와 로고 만들기는 그 값을 쓸 기능이 정해진 뒤에 채운다.
//   store: 저장된 가게 정보, onSave(store): 저장하기
export default function StoreSetup({ store, onSave }) {
  const [name, setName] = useState(store.name)
  const [bio, setBio] = useState(store.bio ?? '')
  const [saved, setSaved] = useState(false)
  const trimmed = name.trim()
  const changed = trimmed !== store.name || bio.trim() !== (store.bio ?? '')

  function save(event) {
    event.preventDefault()
    onSave({ ...store, name: trimmed, bio: bio.trim() })
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

        <label className="field-label" htmlFor="storeBio">
          한 줄 소개 <span className="optional">(선택)</span>
        </label>
        <textarea
          id="storeBio"
          rows={2}
          placeholder="예: 매일 새벽 들여오는 동네 수산입니다. 영업 오전 9시~오후 8시"
          maxLength={MAX_STORE_BIO}
          value={bio}
          onChange={(event) => {
            setBio(event.target.value)
            setSaved(false)
          }}
        />
        <p className="hint">가게 프로필의 이름 아래에 나옵니다. {bio.length} / {MAX_STORE_BIO}자</p>

        <div className="footer">
          <span className="hint" role="status">{saved && !changed ? '저장했습니다.' : ''}</span>
          <button type="submit" className="primary" disabled={!trimmed || !changed}>저장</button>
        </div>
      </form>

      <div className="card placeholder">업종, 분위기 입력 칸이 들어갈 자리</div>
    </section>
  )
}
