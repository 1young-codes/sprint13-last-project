import { useRef, useState } from 'react'

const ACCEPTED = ['image/jpeg', 'image/png']

// 제품 사진 한 장을 끌어다 놓거나 골라서 올리는 칸.
// onChange(file): 올바른 사진이면 파일, 지우면 null
export default function PhotoDrop({ previewUrl, onChange }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')

  function accept(file) {
    if (!file) return
    if (!ACCEPTED.includes(file.type)) {
      setError('jpg나 png 사진만 올릴 수 있습니다.')
      return
    }
    setError('')
    onChange(file)
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragging(false)
    accept(event.dataTransfer.files[0])
  }

  return (
    <div>
      <div
        className={dragging ? 'drop dragging' : 'drop'}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <div className="drop-preview">
            <img src={previewUrl} alt="올린 제품 사진" />
            <button type="button" className="link" onClick={() => onChange(null)}>
              다른 사진으로 바꾸기
            </button>
          </div>
        ) : (
          <>
            <span className="drop-icon" aria-hidden="true">⇪</span>
            <p>사진을 이곳에 끌어다 놓거나 파일을 고르세요</p>
            <button type="button" className="secondary" onClick={() => inputRef.current?.click()}>
              파일 고르기
            </button>
            <p className="hint">jpg, png 한 장. 제품이 잘 보이는 사진이면 됩니다.</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(',')}
          hidden
          onChange={(event) => {
            accept(event.target.files[0])
            event.target.value = ''
          }}
        />
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  )
}
