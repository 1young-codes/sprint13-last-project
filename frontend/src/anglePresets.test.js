import { test } from 'node:test'
import assert from 'node:assert/strict'
import { ANGLE_LABELS, ANGLE_PRESETS, estimateMinutes } from './anglePresets.js'

test('예상 시간: 11/27 측정값(각도 없음 18.3초, 각도 1개 105.1초)보다 짧게 말하지 않는다', () => {
  assert.ok(estimateMinutes(0) * 60 >= 18.3)
  assert.ok(estimateMinutes(1) * 60 >= 105.1)
})

test('예상 시간: 각도를 더 고르면 줄지 않고, 최소 1분이다', () => {
  assert.equal(estimateMinutes(0), 1)
  for (let count = 1; count <= ANGLE_PRESETS.length; count += 1) {
    assert.ok(estimateMinutes(count) >= estimateMinutes(count - 1), `${count}개`)
  }
})

test('각도 이름: 서버 각도 id마다 한국어 이름이 있다', () => {
  for (const preset of ANGLE_PRESETS) {
    assert.match(ANGLE_LABELS[preset.id], /[가-힣]/)
  }
})
