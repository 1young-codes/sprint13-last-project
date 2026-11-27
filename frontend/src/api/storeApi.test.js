import { beforeEach, test } from 'node:test'
import assert from 'node:assert/strict'
import { loadStore, MAX_STORE_NAME, saveStore } from './storeApi.js'

// 브라우저 저장소를 흉내 낸다
beforeEach(() => {
  const data = new Map()
  globalThis.localStorage = {
    getItem: (key) => (data.has(key) ? data.get(key) : null),
    setItem: (key, value) => data.set(key, String(value)),
    removeItem: (key) => data.delete(key),
  }
})

test('가게 이름: 저장한 이름을 다시 불러온다', () => {
  saveStore({ name: '바다수산' })
  assert.equal(loadStore().name, '바다수산')
})

test('가게 이름: 앞뒤 빈칸을 지우고 30자까지만 저장한다', () => {
  assert.equal(saveStore({ name: '  바다수산  ' }).name, '바다수산')
  assert.equal(saveStore({ name: '가'.repeat(40) }).name.length, MAX_STORE_NAME)
})

test('가게 이름: 저장한 적이 없거나 저장소가 망가져도 빈 이름으로 시작한다', () => {
  assert.equal(loadStore().name, '')
  localStorage.setItem('store-sns.store', '{망가진')
  assert.equal(loadStore().name, '')
})
