import { useEffect, useState } from 'react'
import Stepper from '../components/Stepper.jsx'
import PhotoDrop from '../components/PhotoDrop.jsx'
import PostCard from '../components/PostCard.jsx'
import AnglePicker from '../components/AnglePicker.jsx'
import { createPostJob, getPostJob } from '../api/postApi.js'

// 가게 설정 화면이 생기기 전까지 쓰는 임시 가게 이름
const STORE_NAME = '[가게 이름]'
const POLL_MS = 500

// 게시글 만들기 (매번): ① 입력 → ② 결과 → ③ 게시
export default function CreatePost() {
  const [step, setStep] = useState(0)
  const [photo, setPhoto] = useState(null)
  const [photoUrl, setPhotoUrl] = useState('')
  const [description, setDescription] = useState('')
  const [angles, setAngles] = useState([]) // 고른 카메라 각도 id 목록 (비면 원본 그대로)
  const [job, setJob] = useState(null) // { jobId, status, product_photo, usage_example, message }

  // 사진이 바뀌면 미리보기 주소를 새로 만들고, 이전 주소는 정리한다.
  function changePhoto(file) {
    if (photoUrl) URL.revokeObjectURL(photoUrl)
    setPhoto(file)
    setPhotoUrl(file ? URL.createObjectURL(file) : '')
  }

  // 생성 중이면 일정 간격으로 상태를 묻는다.
  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return undefined
    const timer = setTimeout(async () => {
      const next = await getPostJob(job.jobId)
      setJob({ jobId: job.jobId, ...next })
    }, POLL_MS)
    return () => clearTimeout(timer)
  }, [job])

  const canStart = Boolean(photo) && description.trim().length > 0

  async function start() {
    const { jobId } = await createPostJob({ photoUrl, description: description.trim(), angles })
    setJob({ jobId, status: 'queued' })
    setStep(1)
  }

  function restart() {
    changePhoto(null)
    setDescription('')
    setAngles([])
    setJob(null)
    setStep(0)
  }

  return (
    <section className="page">
      <Stepper steps={['입력', '결과', '게시']} current={step} />

      {step === 0 && (
        <>
          <h1>오늘 올릴 제품을 알려주세요</h1>
          <p className="lead">사진 한 장과 설명을 넣으면 게시글 2개를 만듭니다. 하나는 제품 소개, 하나는 활용 예시입니다.</p>
          <div className="card form">
            <label className="field-label">제품 사진</label>
            <PhotoDrop previewUrl={photoUrl} onChange={changePhoto} />

            <label className="field-label" htmlFor="description">제품 설명</label>
            <textarea
              id="description"
              rows={3}
              placeholder="예: 오늘 아침 들어온 자연산 도미, 회와 구이용"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
            <p className="hint">제품 이름과 알리고 싶은 점을 한두 줄로 적어주세요.</p>

            <AnglePicker selected={angles} onChange={setAngles} />
          </div>
          <div className="footer">
            <span className="hint">
              {canStart ? '다음을 누르면 게시글을 만들기 시작합니다. 잠시 걸릴 수 있습니다.' : '사진과 설명을 모두 넣으면 다음으로 갈 수 있습니다.'}
            </span>
            <button type="button" className="primary" disabled={!canStart} onClick={start}>
              다음
            </button>
          </div>
        </>
      )}

      {step === 1 && (
        <ResultStep
          job={job}
          onBack={() => setStep(0)}
          onPublish={() => setStep(2)}
        />
      )}

      {step === 2 && job?.status === 'ready' && (
        <PublishedStep job={job} onMore={restart} />
      )}
    </section>
  )
}

function ResultStep({ job, onBack, onPublish }) {
  const waiting = job.status === 'queued' || job.status === 'running'
  const ready = job.status === 'ready'

  return (
    <>
      <h1>{ready ? '게시글 2개를 만들었습니다' : '게시글을 만들고 있습니다'}</h1>
      <p className="lead">
        {ready ? 'PC에서 게시물을 열었을 때의 모습입니다. 확인한 뒤 게시를 누르세요.' : '보통 1분 안에 끝납니다. 이 화면을 닫지 말고 기다려 주세요.'}
      </p>

      {waiting && (
        <div className="card status" role="status">
          <span className="spinner" aria-hidden="true" />
          {job.status === 'queued' ? '순서를 기다리는 중입니다' : '사진과 문구를 만드는 중입니다'}
        </div>
      )}

      {job.status === 'failed' && (
        <div className="card status error" role="alert">
          게시글을 만들지 못했습니다. {job.message}
        </div>
      )}

      {ready && (
        <>
          <PostCard kind="product" storeName={STORE_NAME} result={job.product_photo} />
          <PostCard kind="usage" storeName={STORE_NAME} result={job.usage_example} />

          <div className="card">
            <label className="field-label" htmlFor="revise">수정 요청</label>
            <div className="row">
              <input id="revise" type="text" placeholder="배경을 더 밝게, 문구는 짧게" disabled />
              <button type="button" className="secondary" disabled>반영</button>
            </div>
            <p className="hint">게시글에 함께 반영됩니다. 하나만 고치려면 그 게시글의 버튼을 쓰세요. (기능 연결 전)</p>
          </div>
        </>
      )}

      <div className="footer">
        <button type="button" className="secondary" onClick={onBack}>이전</button>
        <button type="button" className="primary" disabled={!ready} onClick={onPublish}>게시</button>
      </div>
    </>
  )
}

function PublishedStep({ job, onMore }) {
  return (
    <>
      <div className="card done-banner" role="status">
        <strong>✓ 게시글 2개를 올렸습니다</strong>
        <span>아래는 손님이 가게 프로필에 들어왔을 때 보게 될 모습입니다.</span>
      </div>

      <div className="card profile">
        <header className="profile-head">
          <span className="avatar large">로고</span>
          <div>
            <strong>{STORE_NAME}</strong>
            <p className="hint">[프로필 소개글]</p>
          </div>
        </header>
        <div className="grid">
          <FeedTile label="제품 소개" url={job.product_photo.image_url} />
          <FeedTile label="활용 예시" url={job.usage_example.image_url} />
          <div className="tile empty">다음 게시글이 여기에 쌓입니다</div>
        </div>
      </div>

      <div className="footer end">
        <button type="button" className="secondary" disabled title="인스타그램 연동 여부가 정해지면 연결합니다">
          인스타그램에서 확인하기
        </button>
        <button type="button" className="primary" onClick={onMore}>게시글 더 만들기</button>
      </div>
    </>
  )
}

function FeedTile({ label, url }) {
  return (
    <div className="tile">
      <span className="tile-badge">새 게시물</span>
      {url ? <img src={url} alt={label} /> : <span>[{label} 사진]</span>}
    </div>
  )
}
