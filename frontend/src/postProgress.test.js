import { test } from 'node:test'
import assert from 'node:assert/strict'
import { postProgress } from './postProgress.js'

const PRODUCT = {
  image_url: '/jobs/1/files/product_photo.png',
  angle_images: [
    { angle: 'low_angle', image_url: '/jobs/1/files/low_angle.png', error: null },
    { angle: 'wide_angle', image_url: null, error: '이 각도는 만들지 못했습니다.' },
  ],
}

test('대기 중: 앞에 몇 개가 있는지 알려 준다', () => {
  const result = postProgress({ status: 'queued', queue_position: 2, angles: [] })
  assert.equal(result.waiting, true)
  assert.equal(result.statusText, '순서를 기다리는 중입니다 (앞에 2개)')
  assert.equal(postProgress({ status: 'queued', queue_position: 0 }).statusText, '순서를 기다리는 중입니다')
})

test('만드는 중: 아직 안 끝난 각도와 진행 숫자를 계산한다', () => {
  const job = { status: 'running', angles: ['low_angle', 'wide_angle', 'high_angle'], product_photo: { ...PRODUCT, angle_images: PRODUCT.angle_images.slice(0, 1) } }
  const result = postProgress(job)
  assert.deepEqual(result.pendingAngles, ['wide_angle', 'high_angle'])
  assert.equal(result.finished, 1)
  assert.match(result.statusText, /^각도 사진을 만드는 중입니다 \(1\s\/\s3\)$/)
})

test('각도가 다 끝나면 문구를 만드는 중이라고 알려 준다', () => {
  const job = { status: 'running', angles: ['low_angle', 'wide_angle'], product_photo: PRODUCT }
  assert.equal(postProgress(job).statusText, '문구와 해시태그를 만드는 중입니다')
})

test('완료: 실패한 각도와 체크를 끈 사진은 게시할 사진에서 빠진다', () => {
  const job = { status: 'ready', angles: ['low_angle', 'wide_angle'], product_photo: PRODUCT, usage_example: {} }
  const all = postProgress(job)
  assert.equal(all.ready, true)
  assert.equal(all.waiting, false)
  assert.deepEqual(all.chosen.map((slide) => slide.key), ['original', 'low_angle'])

  const withoutOriginal = postProgress(job, ['original'])
  assert.deepEqual(withoutOriginal.chosen.map((slide) => slide.key), ['low_angle'])
})

test('대표 사진이 아직 없으면 게시할 사진도 없다', () => {
  const result = postProgress({ status: 'running', angles: ['low_angle'] })
  assert.deepEqual(result.chosen, [])
  assert.equal(result.statusText, '사진과 문구를 만드는 중입니다')
})
