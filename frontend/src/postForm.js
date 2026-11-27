// 게시글 만들기 입력값 검사. 화면 모양과 서버 연결이 들어 있지 않아 테스트에서 바로 부를 수 있다.

// 문구 생성 함수(유진님, generate_product_captions)가 받는 값과 같은 칸
export const EMPTY_FORM = { productName: '', unit: '', price: '', usageExample: '', notes: '' }

// 가격은 0보다 큰 정수만 받는다. 쉼표나 '원'을 붙여 써도 숫자만 읽는다.
export function parsePrice(text) {
  const digits = text.replace(/[,\s원]/g, '')
  if (!/^\d+$/.test(digits)) return null
  const value = Number(digits)
  return value > 0 ? value : null
}

// 아직 안 넣은 칸 이름 목록. 비어 있으면 시작할 수 있다.
export function missingFields(photo, form) {
  return [
    !photo && '제품 사진',
    !form.productName.trim() && '품목',
    !form.unit.trim() && '단위',
    parsePrice(form.price) === null && '가격',
    !form.usageExample.trim() && '활용 예시',
  ].filter(Boolean)
}
