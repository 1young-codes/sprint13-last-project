import { ANGLE_LABELS } from './anglePresets.js'

// 캐러셀에 들어갈 사진 목록. 첫 장은 대표(원본) 사진, 그 뒤는 고른 순서대로 각도 사진.
export function postSlides(result, pendingAngles = []) {
  const slides = [{ key: 'original', label: '원본', url: result.image_url }]
  for (const item of result.angle_images ?? []) {
    slides.push({ key: item.angle, label: ANGLE_LABELS[item.angle] ?? item.angle, url: item.image_url, error: item.error })
  }
  for (const angle of pendingAngles) {
    slides.push({ key: angle, label: ANGLE_LABELS[angle] ?? angle, pending: true })
  }
  return slides
}
