import { beforeEach, test } from 'node:test'
import assert from 'node:assert/strict'
import { loadStore, MAX_STORE_BIO, MAX_STORE_NAME, saveStore } from './storeApi.js'

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

test('한 줄 소개: 앞뒤 빈칸을 지우고 150자까지만, 예전에 저장한 정보에는 빈 소개로 채운다', () => {
  assert.equal(saveStore({ name: '바다수산', bio: '  매일 새벽 들여옵니다  ' }).bio, '매일 새벽 들여옵니다')
  assert.equal(saveStore({ name: '바다수산', bio: '가'.repeat(200) }).bio.length, MAX_STORE_BIO)
  localStorage.setItem('store-sns.store', JSON.stringify({ name: '바다수산' }))
  assert.equal(loadStore().bio, '')
})
