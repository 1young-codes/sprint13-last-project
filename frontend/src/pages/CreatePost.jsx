import { useEffect, useState } from 'react'
import Stepper from '../components/Stepper.jsx'
import PhotoDrop from '../components/PhotoDrop.jsx'
import PostCard from '../components/PostCard.jsx'
import AnglePicker from '../components/AnglePicker.jsx'
import { createPostJob, getPostJob, POLL_MS, USE_SAMPLE } from '../api/postApi.js'
import { estimateMinutes } from '../anglePresets.js'
import { postSlides } from '../postSlides.js'

// 가게 설정 화면이 생기기 전까지 쓰는 임시 가게 이름
const STORE_NAME = '[가게 이름]'

// 문구 생성 함수(유진님, generate_product_captions)가 받는 값과 같은 칸
const EMPTY_FORM = { productName: '', unit: '', price: '', usageExample: '', notes: '' }

// 가격은 0보다 큰 정수만 받는다. 쉼표나 '원'을 붙여 써도 숫자만 읽는다.
function parsePrice(text) {
  const digits = text.replace(/[,\s원]/g, '')
  if (!/^\d+$/.test(digits)) return null
  const value = Number(digits)
  return value > 0 ? value : null
}

// 게시글 만들기 (매번): ① 입력 → ② 결과 → ③ 게시
export default function CreatePost() {
  const [step, setStep] = useState(0)
  const [photo, setPhoto] = useState(null)
  const [photoUrl, setPhotoUrl] = useState('')
  const [form, setForm] = useState(EMPTY_FORM) // 품목, 단위, 가격, 활용 예시, 기타
  const [angles, setAngles] = useState([]) // 고른 카메라 각도 id 목록 (비면 원본 그대로)
  const [job, setJob] = useState(null) // { jobId, angles, status, queue_position, product_photo, usage_example, message }
  const [excluded, setExcluded] = useState([]) // 결과에서 체크를 꺼서 뺀 사진 key 목록
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState('') // 서버가 돌려준 한국어 문장

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
      setJob({ jobId: job.jobId, angles: job.angles, ...next })
    }, POLL_MS)
    return () => clearTimeout(timer)
  }, [job])

  function setField(name, value) {
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  const price = parsePrice(form.price)
  const priceError = form.price.trim() !== '' && price === null
  const missing = [
    !photo && '제품 사진',
    !form.productName.trim() && '품목',
    !form.unit.trim() && '단위',
    price === null && '가격',
    !form.usageExample.trim() && '활용 예시',
  ].filter(Boolean)
  const canStart = missing.length === 0

  async function start() {
    setSubmitting(true)
    setSubmitError('')
    try {
      const { jobId } = await createPostJob({
        photo,
        productName: form.productName.trim(),
        unit: form.unit.trim(),
        priceKrw: price,
        usageExample: form.usageExample.trim(),
        notes: form.notes.trim(),
        angles,
      })
      setJob({ jobId, status: 'queued', angles })
      setExcluded([])
      setStep(1)
    } catch (error) {
      setSubmitError(error.message)
    } finally {
      setSubmitting(false)
    }
  }

  function toggleSlide(key) {
    setExcluded((prev) => (prev.includes(key) ? prev.filter((x) => x !== key) : [...prev, key]))
  }

  function restart() {
    changePhoto(null)
    setForm(EMPTY_FORM)
    setAngles([])
    setJob(null)
    setExcluded([])
    setStep(0)
  }

  return (
    <section className="page">
      <Stepper steps={['입력', '결과', '게시']} current={step} />

      {step === 0 && (
        <>
          <h1>오늘 올릴 제품을 알려주세요</h1>
          <p className="lead">사진 한 장과 상품 정보를 넣으면 게시글 2개를 만듭니다. 하나는 제품 소개, 하나는 활용 예시입니다.</p>
          <div className="card form">
            <label className="field-label">제품 사진</label>
            <PhotoDrop previewUrl={photoUrl} onChange={changePhoto} />

            <div className="fields">
              <div className="field">
                <label className="field-label" htmlFor="productName">품목</label>
                <input
                  id="productName"
                  type="text"
                  placeholder="예: 자연산 도미"
                  value={form.productName}
                  onChange={(event) => setField('productName', event.target.value)}
                />
              </div>
              <div className="field">
                <label className="field-label" htmlFor="unit">단위</label>
                <input
                  id="unit"
                  type="text"
                  placeholder="예: 1마리, kg, 500g"
                  value={form.unit}
                  onChange={(event) => setField('unit', event.target.value)}
                />
              </div>
              <div className="field">
                <label className="field-label" htmlFor="price">가격 (원)</label>
                <input
                  id="price"
                  type="text"
                  inputMode="numeric"
                  placeholder="예: 25000"
                  value={form.price}
                  aria-invalid={priceError}
                  onChange={(event) => setField('price', event.target.value)}
                />
                {priceError && <p className="error">숫자로만 적어주세요. 예: 25000</p>}
              </div>
            </div>

            <label className="field-label" htmlFor="usageExample">활용 예시</label>
            <textarea
              id="usageExample"
              rows={2}
              placeholder="예: 회로 먹거나, 소금구이, 맑은 탕으로 끓이기"
              value={form.usageExample}
              onChange={(event) => setField('usageExample', event.target.value)}
            />
            <p className="hint">손님이 이 상품을 어떻게 먹거나 쓰면 좋은지 적어주세요. 활용 예시 게시글에 쓰입니다.</p>

            <label className="field-label" htmlFor="notes">기타 <span className="optional">(선택)</span></label>
            <textarea
              id="notes"
              rows={2}
              placeholder="예: 오늘 아침 직접 받아온 것, '무첨가' 같은 표현은 빼주세요"
              value={form.notes}
              onChange={(event) => setField('notes', event.target.value)}
            />
            <p className="hint">강조하고 싶은 점이나 넣지 말았으면 하는 표현을 자유롭게 적어주세요.</p>

            <AnglePicker selected={angles} onChange={setAngles} />
          </div>
          {submitError && <div className="card status error" role="alert">{submitError}</div>}
          <div className="footer">
            <span className="hint">
              {canStart ? `다음을 누르면 게시글을 만들기 시작합니다. 약 ${estimateMinutes(angles.length)}분 걸립니다.` : `${missing.join(', ')}을(를) 넣으면 다음으로 갈 수 있습니다.`}
            </span>
            <button type="button" className="primary" disabled={!canStart || submitting} onClick={start}>
              {submitting ? '보내는 중…' : '다음'}
            </button>
          </div>
        </>
      )}

      {step === 1 && (
        <ResultStep
          job={job}
          excluded={excluded}
          onToggle={toggleSlide}
          onBack={() => setStep(0)}
          onPublish={() => setStep(2)}
        />
      )}

      {step === 2 && job?.status === 'ready' && (
        <PublishedStep job={job} excluded={excluded} onMore={restart} />
      )}
    </section>
  )
}

function ResultStep({ job, excluded, onToggle, onBack, onPublish }) {
  const waiting = job.status === 'queued' || job.status === 'running'
  const ready = job.status === 'ready'
  const product = job.product_photo
  const requested = job.angles ?? []
  const finished = product?.angle_images?.length ?? 0
  const pendingAngles = requested.slice(finished)
  const chosen = product ? postSlides(product).filter((slide) => slide.url && !excluded.includes(slide.key)) : []

  let statusText = '사진과 문구를 만드는 중입니다'
  if (job.status === 'queued') {
    statusText = job.queue_position > 0 ? `순서를 기다리는 중입니다 (앞에 ${job.queue_position}개)` : '순서를 기다리는 중입니다'
  } else if (product && pendingAngles.length > 0) {
    statusText = `각도 사진을 만드는 중입니다 (${finished} / ${requested.length})`
  } else if (product) {
    statusText = '문구와 해시태그를 만드는 중입니다'
  }

  return (
    <>
      <h1>{ready ? '게시글 2개를 만들었습니다' : '게시글을 만들고 있습니다'}</h1>
      <p className="lead">
        {ready
          ? 'PC에서 게시물을 열었을 때의 모습입니다. 사진을 넘겨 보고, 어색한 사진은 체크를 꺼서 뺀 뒤 게시를 누르세요.'
          : `약 ${estimateMinutes(requested.length)}분 걸립니다. 다 된 사진부터 아래에 나옵니다. 이 화면을 닫지 말고 기다려 주세요.`}
      </p>
      {USE_SAMPLE && <p className="hint">지금은 서버 없이 예시 결과를 보여줍니다.</p>}

      {waiting && (
        <div className="card status" role="status">
          <span className="spinner" aria-hidden="true" />
          {statusText}
        </div>
      )}

      {job.status === 'failed' && (
        <div className="card status error" role="alert">
          게시글을 만들지 못했습니다. {job.message}
        </div>
      )}

      {product && (
        <PostCard
          kind="product"
          storeName={STORE_NAME}
          result={product}
          pendingAngles={waiting ? pendingAngles : []}
          excluded={excluded}
          onToggle={onToggle}
        />
      )}
      {job.usage_example && <PostCard kind="usage" storeName={STORE_NAME} result={job.usage_example} />}

      {ready && (
        <div className="card">
          <label className="field-label" htmlFor="revise">수정 요청</label>
          <div className="row">
            <input id="revise" type="text" placeholder="배경을 더 밝게, 문구는 짧게" disabled />
            <button type="button" className="secondary" disabled>반영</button>
          </div>
          <p className="hint">게시글에 함께 반영됩니다. 하나만 고치려면 그 게시글의 버튼을 쓰세요. (기능 연결 전)</p>
        </div>
      )}

      {ready && chosen.length === 0 && (
        <p className="error">제품 소개 사진을 한 장 이상 넣어야 게시할 수 있습니다.</p>
      )}
      <div className="footer">
        <button type="button" className="secondary" onClick={onBack}>이전</button>
        <button type="button" className="primary" disabled={!ready || chosen.length === 0} onClick={onPublish}>게시</button>
      </div>
    </>
  )
}

function PublishedStep({ job, excluded, onMore }) {
  const chosen = postSlides(job.product_photo).filter((slide) => slide.url && !excluded.includes(slide.key))

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
          <FeedTile label="제품 소개" url={chosen[0]?.url} count={chosen.length} />
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

function FeedTile({ label, url, count = 1 }) {
  return (
    <div className="tile">
      <span className="tile-badge">새 게시물</span>
      {count > 1 && <span className="tile-many" title={`사진 ${count}장`}>⧉ {count}</span>}
      {url ? <img src={url} alt={label} /> : <span>[{label} 사진]</span>}
    </div>
  )
}
