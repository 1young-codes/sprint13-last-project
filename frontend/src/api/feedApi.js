// 내 피드: 게시한 게시글 목록.
// 인스타그램 실제 연결과 서버 저장이 정해지기 전까지는 이 브라우저에만 저장한다.
// 서버에 저장하게 되면 이 파일만 바꾼다. 화면(컴퓨터, 휴대폰)은 그대로 쓴다.
//
// 저장하는 게시글 하나의 모양 (최신이 앞):
//   { id, kind: 'product' | 'usage', productName, createdAt, images: [사진 주소...], caption, hashtags: [...] }
// 사진은 작게 줄여서 글자(data 주소)로 저장한다. 서버 사진 주소나 예시 결과의 임시 주소는
// 서버가 다시 켜지거나 새로 고침하면 사라지기 때문이다.

const STORAGE_KEY = 'store-sns.feed'
const MAX_POSTS = 30 // 브라우저 저장 공간(약 5MB)을 넘지 않게 최근 것만 둔다
const THUMB_SIZE = 480 // 피드에 저장하는 사진의 긴 변(픽셀)

export function loadFeed() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    return Array.isArray(saved) ? saved : []
  } catch {
    return []
  }
}

function writeFeed(posts) {
  // 저장 공간이 모자라면 오래된 것부터 빼고 다시 저장한다.
  for (let kept = posts.slice(0, MAX_POSTS); kept.length > 0; kept = kept.slice(0, -1)) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(kept))
      return kept
    } catch {
      // 다음 바퀴에서 하나 덜 저장한다
    }
  }
  return posts.slice(0, MAX_POSTS) // 저장이 안 되는 브라우저: 이번 방문 동안만 보인다
}

// 사진 주소를 작은 사진(data 주소)으로 바꾼다. 못 바꾸면 원래 주소를 그대로 쓴다.
function toThumbnail(url) {
  if (!url || url.startsWith('data:')) return Promise.resolve(url)
  return new Promise((resolve) => {
    const img = new Image()
    img.crossOrigin = 'anonymous' // 서버 사진을 그림판(canvas)에 옮기려면 필요하다
    img.onload = () => {
      try {
        const scale = Math.min(1, THUMB_SIZE / Math.max(img.naturalWidth, img.naturalHeight))
        const canvas = document.createElement('canvas')
        canvas.width = Math.round(img.naturalWidth * scale)
        canvas.height = Math.round(img.naturalHeight * scale)
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height)
        resolve(canvas.toDataURL('image/jpeg', 0.8))
      } catch {
        resolve(url)
      }
    }
    img.onerror = () => resolve(url)
    img.src = url
  })
}

// 게시글 2개(제품 소개, 활용 예시)를 피드 맨 앞에 넣고, 새 목록을 돌려준다.
//   productName: 품목, productImages: 고른 제품 소개 사진 주소들, product/usage: 서버 결과의 product_photo, usage_example
export async function publishPosts({ productName, productImages, product, usage }) {
  const createdAt = new Date().toISOString()
  const stamp = Date.now()
  const [productThumbs, usageThumb] = await Promise.all([
    Promise.all(productImages.map(toThumbnail)),
    toThumbnail(usage?.image_url),
  ])
  const added = [
    // 인스타그램처럼 나중에 올린 것이 앞에 온다: 활용 예시가 맨 앞, 제품 소개가 그다음
    {
      id: `${stamp}-usage`,
      kind: 'usage',
      productName,
      createdAt,
      images: usageThumb ? [usageThumb] : [],
      caption: usage?.caption ?? '',
      hashtags: usage?.hashtags ?? [],
    },
    {
      id: `${stamp}-product`,
      kind: 'product',
      productName,
      createdAt,
      images: productThumbs.filter(Boolean),
      caption: product?.caption ?? '',
      hashtags: product?.hashtags ?? [],
    },
  ]
  return writeFeed([...added, ...loadFeed()])
}

export function clearFeed() {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    // 지울 것이 없으면 그대로 둔다
  }
  return []
}
