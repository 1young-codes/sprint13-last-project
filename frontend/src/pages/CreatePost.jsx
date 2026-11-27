import Stepper from '../components/Stepper.jsx'
import PhotoDrop from '../components/PhotoDrop.jsx'
import PostCard from '../components/PostCard.jsx'
import AnglePicker from '../components/AnglePicker.jsx'
import { FeedGrid, ProfileHead } from '../components/Feed.jsx'
import { USE_SAMPLE } from '../api/postApi.js'
import { estimateMinutes } from '../anglePresets.js'
import { useCreatePost } from '../hooks/useCreatePost.js'
import { postProgress } from '../postProgress.js'

// 게시글 만들기 (매번): ① 입력 → ② 결과 → ③ 게시
// 이 파일은 컴퓨터 화면의 "모양"만 맡는다. 입력 검사, 상태 묻기, 사진 빼기 같은 동작은
// hooks/useCreatePost.js에 있어서, 휴대폰 전용 화면을 만들어도 그대로 쓸 수 있다.
//   storeName: 게시글 위쪽과 가게 프로필에 나오는 이름 (가게 설정에서 넣는다)
//   feed: 게시한 게시글 목록, onPublished(feed): 게시해서 피드가 바뀌었을 때, onGoTo(tab): 다른 탭으로
export default function CreatePost({ storeName, feed = [], onPublished, onGoTo }) {
  const {
    step,
    photoUrl,
    changePhoto,
    form,
    setField,
    angles,
    setAngles,
    priceError,
    missing,
    canStart,
    submitting,
    submitError,
    start,
    job,
    excluded,
    toggleSlide,
    backToInput,
    publishing,
    publish,
    restart,
  } = useCreatePost({ onPublished })

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
          storeName={storeName}
          excluded={excluded}
          onToggle={toggleSlide}
          onBack={backToInput}
          publishing={publishing}
          onPublish={publish}
        />
      )}

      {step === 2 && job?.status === 'ready' && (
        <PublishedStep storeName={storeName} feed={feed} onMore={restart} onGoTo={onGoTo} />
      )}
    </section>
  )
}

function ResultStep({ job, storeName, excluded, onToggle, onBack, publishing, onPublish }) {
  const { waiting, ready, product, requested, pendingAngles, chosen, statusText } = postProgress(job, excluded)

  return (
    <>
      <h1>{ready ? '게시글 2개를 만들었습니다' : '게시글을 만들고 있습니다'}</h1>
      <p className="lead">
        {ready
          ? '인스타그램에서 게시물을 열었을 때의 모습입니다. 사진을 넘겨 보고, 어색한 사진은 체크를 꺼서 뺀 뒤 게시를 누르세요.'
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
          {job.message || '게시글을 만들지 못했습니다. 다시 시도해 주세요.'}
        </div>
      )}

      {product && (
        <PostCard
          kind="product"
          storeName={storeName}
          result={product}
          pendingAngles={waiting ? pendingAngles : []}
          excluded={excluded}
          onToggle={onToggle}
        />
      )}
      {job.usage_example && <PostCard kind="usage" storeName={storeName} result={job.usage_example} />}

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
        <button type="button" className="primary" disabled={!ready || chosen.length === 0 || publishing} onClick={onPublish}>
          {publishing ? '올리는 중…' : '게시'}
        </button>
      </div>
    </>
  )
}

// 게시 뒤: 방금 올린 2개에 "새 게시물" 표시, 그 뒤에 예전 게시글이 이어진다 (최대 4개 더)
const PREVIOUS_SHOWN = 4

function PublishedStep({ storeName, feed, onMore, onGoTo }) {
  const shown = feed.slice(0, 2 + PREVIOUS_SHOWN)

  return (
    <>
      <div className="card done-banner" role="status">
        <strong>✓ 게시글 2개를 올렸습니다</strong>
        <span>아래는 손님이 가게 프로필에 들어왔을 때 보게 될 모습입니다.</span>
      </div>

      <div className="card profile">
        <ProfileHead storeName={storeName} count={feed.length} />
        <FeedGrid
          posts={shown}
          newCount={2}
          tail={
            feed.length > shown.length ? (
              <button type="button" className="tile empty tile-button" onClick={() => onGoTo?.('feed')}>
                내 피드에서 {feed.length - shown.length}개 더 보기
              </button>
            ) : (
              <div className="tile empty">다음 게시글이 여기에 쌓입니다</div>
            )
          }
        />
      </div>

      <div className="footer end">
        <button type="button" className="secondary" disabled title="인스타그램 연동 여부가 정해지면 연결합니다">
          인스타그램에서 확인하기
        </button>
        <button type="button" className="secondary" onClick={() => onGoTo?.('feed')}>내 피드 보기</button>
        <button type="button" className="primary" onClick={onMore}>게시글 더 만들기</button>
      </div>
    </>
  )
}
