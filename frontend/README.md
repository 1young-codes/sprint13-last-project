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

## 폴더

| 경로 | 내용 |
| --- | --- |
| `src/App.jsx` | 위쪽 탭과 탭 전환 |
| `src/pages/` | 탭별 화면 (가게 설정, 게시글 만들기, 내 피드) |
| `src/components/` | 여러 화면에서 같이 쓰는 부품 (단계 표시 등) |
| `src/index.css` | 색, 간격, 글꼴 |
| `app.py` | 예전 Streamlit 결과 확인용 임시 화면 (React 화면이 완성되면 정리) |
