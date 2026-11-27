import { useEffect, useState } from 'react'
import { createPostJob, getPostJob, POLL_MS } from '../api/postApi.js'
import { publishPosts } from '../api/feedApi.js'
import { EMPTY_FORM, missingFields, parsePrice } from '../postForm.js'
import { postProgress } from '../postProgress.js'

// 게시글 만들기의 "동작"만 모은 곳. 화면 모양은 들어 있지 않다.
// 컴퓨터 화면(pages/CreatePost.jsx)과 나중에 만들 휴대폰 전용 화면이 이것을 같이 쓴다.
//
// 흐름: ① 입력 → start() → ② 결과 (2초마다 상태를 묻는다) → publish() → ③ 게시 → restart()
//   onPublished(feed): 게시해서 피드가 바뀌었을 때 새 피드 목록을 받는다

export function useCreatePost({ onPublished } = {}) {
  const [step, setStep] = useState(0) // 0 입력, 1 결과, 2 게시
  const [photo, setPhoto] = useState(null)
  const [photoUrl, setPhotoUrl] = useState('')
  const [form, setForm] = useState(EMPTY_FORM) // 품목, 단위, 가격, 활용 예시, 기타
  const [angles, setAngles] = useState([]) // 고른 카메라 각도 id 목록 (비면 원본 그대로)
  const [job, setJob] = useState(null) // { jobId, angles, status, queue_position, product_photo, usage_example, message }
  const [excluded, setExcluded] = useState([]) // 결과에서 체크를 꺼서 뺀 사진 key 목록
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState('') // 서버가 돌려준 한국어 문장
  const [publishing, setPublishing] = useState(false)

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
  const missing = missingFields(photo, form)
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

  // 고른 사진과 문구를 피드에 올리고 ③ 게시로 넘어간다.
  async function publish() {
    setPublishing(true)
    try {
      const { chosen } = postProgress(job, excluded)
      const feed = await publishPosts({
        productName: form.productName.trim(),
        productImages: chosen.map((slide) => slide.url),
        product: job.product_photo,
        usage: job.usage_example,
      })
      onPublished?.(feed)
      setStep(2)
    } finally {
      setPublishing(false)
    }
  }

  function restart() {
    changePhoto(null)
    setForm(EMPTY_FORM)
    setAngles([])
    setJob(null)
    setExcluded([])
    setStep(0)
  }

  return {
    step,
    // ① 입력
    photo,
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
    // ② 결과
    job,
    excluded,
    toggleSlide,
    backToInput: () => setStep(0),
    publishing,
    publish,
    // ③ 게시
    restart,
  }
}
