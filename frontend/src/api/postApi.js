// 게시글 생성 서버와 주고받는 부분.
// 서버가 아직 없어서 지금은 샘플 결과를 돌려준다. 서버가 생기면 이 파일만 바꾼다.
//
// 서빙 담당과 합의한 흐름(11/23 제안):
// 1. 요청을 보내면 작업 번호(jobId)를 받는다.
// 2. 작업 번호로 상태를 묻는다: queued(대기) / running(생성 중) / ready(완료) / failed(실패)
// 3. 결과는 product_photo(실물 소개)와 usage_example(활용 예시)로 나뉘고,
//    각각 이미지 주소(image_url), 문구(caption), 해시태그(hashtags), 실패 사유(error)를 따로 가진다.

const SAMPLE_DELAY_MS = 1500

const jobs = new Map()

// 보내는 값 (문구 생성 함수 generate_product_captions의 입력과 같다)
//   productName: 품목, unit: 단위, priceKrw: 가격(0보다 큰 정수), usageExample: 활용 예시, notes: 기타(선택)
//   angles: 고른 카메라 각도 id 목록. 비어 있으면 원본 각도 그대로 만든다.
export async function createPostJob({ photoUrl, productName, unit, priceKrw, usageExample, notes = '', angles = [] }) {
  const jobId = `sample-${Date.now()}`
  jobs.set(jobId, { startedAt: Date.now(), photoUrl, productName, unit, priceKrw, usageExample, notes, angles })
  return { jobId }
}

export async function getPostJob(jobId) {
  const job = jobs.get(jobId)
  if (!job) {
    return { status: 'failed', message: '작업을 찾을 수 없습니다.' }
  }
  if (Date.now() - job.startedAt < SAMPLE_DELAY_MS) {
    return { status: 'running' }
  }
  return {
    status: 'ready',
    product_photo: {
      // 실물 소개는 원본 제품을 보존하므로, 샘플에서는 올린 사진을 그대로 보여준다.
      image_url: job.photoUrl,
      caption: '[생성된 문구가 여기에 나옵니다. 문구가 길어도 이 칸 안에서 끝까지 볼 수 있습니다.]',
      hashtags: ['#해시태그', '#해시태그', '#해시태그'],
      error: null,
    },
    usage_example: {
      image_url: null,
      caption: '[생성된 문구가 여기에 나옵니다. 제품을 어떻게 쓰거나 즐기는지 보여주는 글입니다.]',
      hashtags: ['#해시태그', '#해시태그', '#해시태그'],
      error: null,
    },
  }
}
