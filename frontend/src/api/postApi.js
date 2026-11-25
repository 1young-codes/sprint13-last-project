// 게시글 만들기 서버와 주고받는 부분. 주소와 값은 docs/api-spec.md를 따른다.
//
// 흐름:
// 1. 사진과 입력을 보내면 작업 번호(job_id)를 받는다.
// 2. 작업 번호로 상태를 묻는다: queued(대기) / running(만드는 중) / ready(완료) / failed(실패)
// 3. running부터 product_photo가 들어 있고, 각도 사진(angle_images)은 끝난 것부터 하나씩 늘어난다.
//    usage_example은 ready일 때만 들어 있다.
//
// 서버 주소는 frontend/.env.local의 VITE_API_URL로 정한다 (예: http://localhost:8003).
// 비워 두면 서버 없이 문서 모양 그대로의 예시 결과를 돌려준다.

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
export const USE_SAMPLE = API_URL === ''

// 문서: 상태는 2초마다 묻는다. 예시 결과는 빨리 보려고 짧게 묻는다.
export const POLL_MS = USE_SAMPLE ? 500 : 2000

const CONNECTION_ERROR = '서버에 연결하지 못했습니다. 잠시 뒤 다시 시도해 주세요.'

// 서버가 돌려준 이미지 주소(/jobs/...)를 서버 주소 뒤에 붙인다.
function absolute(url) {
  if (!url || USE_SAMPLE) return url
  return `${API_URL}${url}`
}

function withAbsoluteUrls(post) {
  if (!post) return post
  return {
    ...post,
    image_url: absolute(post.image_url),
    angle_images: post.angle_images?.map((item) => ({ ...item, image_url: absolute(item.image_url) })),
  }
}

// 서버가 돌려준 한국어 문장(detail)을 꺼낸다. 없으면 기본 문장.
async function errorMessage(response) {
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') return body.detail
  } catch {
    // 본문이 JSON이 아니면 기본 문장을 쓴다
  }
  return '게시글을 만들지 못했습니다. 다시 시도해 주세요.'
}

// 보내는 값 (문구 생성 함수 generate_product_captions의 입력과 같다)
//   photo: 사진 파일, productName: 품목, unit: 단위, priceKrw: 가격(0보다 큰 정수),
//   usageExample: 활용 예시, notes: 기타(선택), angles: 고른 각도 id 목록(비면 원본 각도 그대로)
// 돌려주는 값: { jobId }. 입력이 잘못되면 서버의 한국어 문장을 담은 Error를 던진다.
export async function createPostJob({ photo, productName, unit, priceKrw, usageExample, notes = '', angles = [] }) {
  if (USE_SAMPLE) return createSampleJob({ photo, productName, angles })

  const form = new FormData()
  form.append('photo', photo)
  form.append('product_name', productName)
  form.append('unit', unit)
  form.append('price_krw', String(priceKrw))
  form.append('usage_example', usageExample)
  form.append('notes', notes)
  angles.forEach((angle) => form.append('angles', angle))

  let response
  try {
    response = await fetch(`${API_URL}/jobs`, { method: 'POST', body: form })
  } catch {
    throw new Error(CONNECTION_ERROR)
  }
  if (!response.ok) throw new Error(await errorMessage(response))
  const { job_id: jobId } = await response.json()
  return { jobId }
}

// 돌려주는 값: 문서의 GET /jobs/{job_id} 응답. 이미지 주소는 바로 <img src>에 넣을 수 있게 바꿔 둔다.
export async function getPostJob(jobId) {
  if (USE_SAMPLE) return getSampleJob(jobId)

  let response
  try {
    response = await fetch(`${API_URL}/jobs/${encodeURIComponent(jobId)}`)
  } catch {
    return { status: 'failed', message: CONNECTION_ERROR }
  }
  if (response.status === 404) {
    return { status: 'failed', message: '작업을 찾을 수 없습니다. 서버가 다시 켜졌을 수 있습니다.' }
  }
  if (!response.ok) return { status: 'failed', message: await errorMessage(response) }

  const job = await response.json()
  return {
    ...job,
    product_photo: withAbsoluteUrls(job.product_photo),
    usage_example: withAbsoluteUrls(job.usage_example),
  }
}

// ---- 서버 없이 쓰는 예시 결과 (문서의 응답 모양과 순서를 그대로 흉내 낸다) ----

const SAMPLE_QUEUE_MS = 800 // 순서 기다리기
const SAMPLE_ANGLE_MS = 1200 // 각도 한 장
const SAMPLE_TEXT_MS = 800 // 문구와 해시태그
const sampleJobs = new Map()

function createSampleJob({ photo, productName, angles }) {
  const jobId = `sample-${Date.now()}`
  sampleJobs.set(jobId, { startedAt: Date.now(), photoUrl: URL.createObjectURL(photo), productName, angles })
  return { jobId }
}

function getSampleJob(jobId) {
  const job = sampleJobs.get(jobId)
  if (!job) return { status: 'failed', message: '작업을 찾을 수 없습니다.' }

  const elapsed = Date.now() - job.startedAt
  const base = { job_id: jobId, message: '' }
  if (elapsed < SAMPLE_QUEUE_MS) {
    return { ...base, status: 'queued', queue_position: 1, product_photo: null, usage_example: null }
  }

  // 각도는 고른 순서대로 끝난다. 예시에서는 원본 사진을 각도 사진 자리에 넣고, 마지막 각도 하나는 실패로 보여준다.
  const done = Math.min(job.angles.length, Math.floor((elapsed - SAMPLE_QUEUE_MS) / SAMPLE_ANGLE_MS))
  const angleImages = job.angles.slice(0, done).map((angle, index) => {
    const failed = job.angles.length > 1 && index === job.angles.length - 1
    return failed
      ? { angle, image_url: null, error: '이 각도는 만들지 못했습니다.' }
      : { angle, image_url: job.photoUrl, error: null }
  })
  const ready = elapsed >= SAMPLE_QUEUE_MS + job.angles.length * SAMPLE_ANGLE_MS + SAMPLE_TEXT_MS
  const product = {
    image_url: job.photoUrl,
    angle_images: angleImages,
    caption: ready ? `[예시 문구] ${job.productName} 소개 문구가 여기에 나옵니다. 문구가 길어도 이 칸 안에서 끝까지 볼 수 있습니다.` : '',
    hashtags: ready ? ['#예시해시태그', '#제품소개'] : [],
    error: null,
  }
  if (!ready) {
    return { ...base, status: 'running', queue_position: null, product_photo: product, usage_example: null }
  }
  return {
    ...base,
    status: 'ready',
    queue_position: null,
    product_photo: product,
    usage_example: {
      image_url: null,
      caption: `[예시 문구] ${job.productName}을(를) 어떻게 쓰거나 즐기는지 보여주는 글입니다.`,
      hashtags: ['#활용예시'],
      error: null,
    },
  }
}
