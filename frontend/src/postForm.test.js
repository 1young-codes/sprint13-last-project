import { test } from 'node:test'
import assert from 'node:assert/strict'
import { EMPTY_FORM, missingFields, parsePrice } from './postForm.js'

test('가격: 쉼표, 빈칸, "원"을 붙여 써도 숫자만 읽는다', () => {
  assert.equal(parsePrice('25000'), 25000)
  assert.equal(parsePrice('25,000원'), 25000)
  assert.equal(parsePrice(' 1 2 000 '), 12000)
})

test('가격: 0, 음수, 소수, 글자는 받지 않는다', () => {
  for (const text of ['', '0', '-500', '12.5', '만원', '2만']) {
    assert.equal(parsePrice(text), null, text)
  }
})

const FILLED = { productName: '자연산 도미', unit: '1마리', price: '25000', usageExample: '회, 소금구이', notes: '' }

test('빠진 칸: 다 넣으면 비어 있고, 기타는 비어도 된다', () => {
  assert.deepEqual(missingFields({ name: 'domi.jpg' }, FILLED), [])
})

test('빠진 칸: 화면에 보이는 이름으로 순서대로 알려 준다', () => {
  assert.deepEqual(missingFields(null, EMPTY_FORM), ['제품 사진', '품목', '단위', '가격', '활용 예시'])
})

test('빠진 칸: 빈칸만 넣거나 가격이 잘못되면 빠진 것으로 본다', () => {
  const form = { ...FILLED, productName: '   ', price: '0원' }
  assert.deepEqual(missingFields({ name: 'domi.jpg' }, form), ['품목', '가격'])
})
