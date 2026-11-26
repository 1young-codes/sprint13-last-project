# 화면 (React)

사장님이 쓰는 화면입니다. 위쪽 탭은 가게 설정, 게시글 만들기, 내 피드 세 개입니다.

## 실행

Node.js가 필요합니다.

```bash
cd frontend
npm install      # 처음 한 번
npm run dev      # http://localhost:8504
```

포트 8504는 VM 사용 규칙에서 화면 담당에게 배정된 번호입니다.

## 서버 연결

게시글 만들기는 서빙 담당의 API 서버를 부릅니다. 주소와 값은 [docs/api-spec.md](../docs/api-spec.md)를 따릅니다.

1. VM에 SSH 터널을 엽니다. 켜고 끄는 방법과 터널 명령은 `serving/README.md`에 있습니다.
2. `frontend/.env.local` 파일을 만들고 아래 한 줄을 적습니다. 이 파일은 GitHub에 올라가지 않습니다.

   ```
   VITE_API_URL=http://localhost:8003
   ```

3. `npm run dev`를 다시 켭니다.

`.env.local`이 없거나 비어 있으면 서버 없이 **예시 결과**로 돌아갑니다. 순서 기다리기, 각도 사진이 한 장씩 늘어나기, 각도 하나 실패, 문구와 해시태그가 나중에 채워지기를 문서 모양 그대로 흉내 냅니다.

## 폴더

| 경로 | 내용 |
| --- | --- |
| `src/App.jsx` | 위쪽 탭과 탭 전환 |
| `src/pages/` | 탭별 화면 (가게 설정, 게시글 만들기, 내 피드) |
| `src/components/` | 여러 화면에서 같이 쓰는 부품 (단계 표시, 게시글 카드, 각도 선택 등) |
| `src/api/` | 서버와 주고받는 부분 (`postApi.js`: 게시글 만들기, `authApi.js`: 계정 연결 예시) |
| `src/anglePresets.js` | 카메라 각도 13개와 예상 시간 계산 |
| `src/index.css` | 색, 간격, 글꼴 |
| `app.py` | 예전 Streamlit 결과 확인용 임시 화면 (React 화면이 완성되면 정리) |
