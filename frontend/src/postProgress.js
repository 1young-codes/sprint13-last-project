import { postSlides } from './postSlides.js'

// 서버가 돌려준 작업 상태(job)를 화면에 보여 줄 값으로 바꾼다. 화면 모양은 들어 있지 않다.
// 컴퓨터 화면과 휴대폰 화면이 같이 쓴다.
//   excluded: 결과에서 체크를 꺼서 뺀 사진 key 목록
export function postProgress(job, excluded = []) {
  const waiting = job.status === 'queued' || job.status === 'running'
  const ready = job.status === 'ready'
  const product = job.product_photo
  const requested = job.angles ?? []
  const finished = product?.angle_images?.length ?? 0
  const pendingAngles = requested.slice(finished) // 아직 만드는 중인 각도
  // 게시할 제품 소개 사진: 다 만들어졌고 체크를 끄지 않은 것
  const chosen = product ? postSlides(product).filter((slide) => slide.url && !excluded.includes(slide.key)) : []

  let statusText = '사진과 문구를 만드는 중입니다'
  if (job.status === 'queued') {
    statusText = job.queue_position > 0 ? `순서를 기다리는 중입니다 (앞에 ${job.queue_position}개)` : '순서를 기다리는 중입니다'
  } else if (product && pendingAngles.length > 0) {
    statusText = `각도 사진을 만드는 중입니다 (${finished} / ${requested.length})`
  } else if (product) {
    statusText = '문구와 해시태그를 만드는 중입니다'
  }

  return { waiting, ready, product, requested, finished, pendingAngles, chosen, statusText }
}
