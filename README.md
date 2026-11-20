# 소상공인 광고 콘텐츠 생성 서비스 (가제)

<!-- 주제가 확정되면 제목과 아래 한 줄 소개를 바꿉니다. -->
생성형 AI로 소상공인이 광고 콘텐츠를 손쉽게 만들 수 있게 돕는 서비스입니다.

> Codeit AI Engineer Bootcamp 13기 고급 프로젝트 (1팀)  
> Project Period: **2026.11.__ ~ 2026.12.18** <!-- 시작일을 채웁니다. -->

## 📌 Project Overview

<!-- 주제 확정 후 작성: 누가, 무엇을 넣으면, 무엇이 나오는가를 한 문장으로 적습니다. -->

- 대상: 광고와 콘텐츠 제작에 인력과 예산이 부족한 소상공인
- 목표: 디자인 경험이 없어도 광고 콘텐츠를 만들 수 있는 서비스
- 메인 기능: (확정 후 작성)

## 👥 Team

| Name | Role | GitHub | 협업 일지 |
| --- | --- | --- | --- |
| 이원영 | (확정 후 작성) | [@1young-codes](https://github.com/1young-codes) | [원영님 협업일지](https://docs.google.com/document/d/1unQFx06byuoRJ_OBuQfDLzCAtA95AdPUmt40sE_25mc/edit?tab=t.ki2tnxupzl55) |
| 권유진 | (확정 후 작성) | (GitHub 아이디) | [유진님 협업일지](https://docs.google.com/document/d/1unQFx06byuoRJ_OBuQfDLzCAtA95AdPUmt40sE_25mc/edit?tab=t.txte2sa0ikx) |
| 김민기 | (확정 후 작성) | (GitHub 아이디) | [민기님 협업일지](https://docs.google.com/document/d/1unQFx06byuoRJ_OBuQfDLzCAtA95AdPUmt40sE_25mc/edit?tab=t.ij7ou9e1jpjb) |
| 나상훈 | (확정 후 작성) | (GitHub 아이디) | [상훈님 협업일지](https://docs.google.com/document/d/1unQFx06byuoRJ_OBuQfDLzCAtA95AdPUmt40sE_25mc/edit?tab=t.ugq6nwqqcb0y) |
| 박단비 | (확정 후 작성) | (GitHub 아이디) | [단비님 협업일지](https://docs.google.com/document/d/1unQFx06byuoRJ_OBuQfDLzCAtA95AdPUmt40sE_25mc/edit?tab=t.f8aooc7qr42y) |

### 회의록 및 일정

- [데일리 스크럼 기록](https://docs.google.com/document/d/1unQFx06byuoRJ_OBuQfDLzCAtA95AdPUmt40sE_25mc/edit?tab=t.2qqo9r9i612o)
- [깃 가이드](https://docs.google.com/document/d/1unQFx06byuoRJ_OBuQfDLzCAtA95AdPUmt40sE_25mc/edit?tab=t.on1kv9xdkp27)
- [Jira 보드](https://wonyounglee226.atlassian.net/jira/software/projects/AD/boards/1)

## 🗂️ Project Structure

<!-- 폴더 구조가 정해지면 실제 구조로 바꿉니다. 아래는 계획입니다. -->

```text
sprint13-last-project/
├── frontend/            # 화면
├── backend/             # API 서버
├── models/              # 프롬프트, 생성 파이프라인, 학습 코드
├── experiments/         # 노트북과 실험 기록
├── tests/               # 동작 확인 코드
├── docs/                # 보고서, 다이어그램, PR 전후 이미지
├── docker-compose.yml
├── requirements.txt
├── .env.example         # 필요한 키 이름만 적은 견본
└── README.md
```

모델 가중치, 데이터, 생성 결과물, `.env`는 Git에 올리지 않습니다.

## 🛠️ Development Environment

| Category | Stack |
| --- | --- |
| Language | Python |
| Model | (확정 후 작성) |
| Framework | (확정 후 작성) |
| Environment | GCP VM (g2-standard-4, NVIDIA L4) |
| Project Management | Jira |
| Version Control | Git, GitHub |

## 🌿 Branch Rule

- `main`: 항상 실행 가능한 안정 버전. `dev`에서 통합 PR로만 갱신합니다.
- `dev`: 기능을 모아 확인하는 브랜치. 작업 브랜치는 `dev`에서 만들고 `dev`로 PR을 올립니다.
- 작업 브랜치: `유형/작업명` (예: `feature/upload-api`)
- 커밋: `유형: 변경 내용` (예: `feature: AD-12 업로드 엔드포인트 추가`)

자세한 규칙은 [깃 가이드](https://docs.google.com/document/d/1unQFx06byuoRJ_OBuQfDLzCAtA95AdPUmt40sE_25mc/edit?tab=t.on1kv9xdkp27)와 `.agent-git/README.md`를 봅니다.

## 🚀 Getting Started

<!-- 실행할 코드가 생기면 작성: 저장소 복제, .env 준비, 실행 명령, 접속 주소 -->

## 🧪 Experiments

<!-- 모델 비교와 설정 실험을 표로 기록합니다: 날짜, 실험, 모델, 바꾼 것, 결과 -->

## 🏆 Results

<!-- 최종 결과물 예시 이미지와 평가 결과를 적습니다. -->

## 📎 Reports

- 보고서 PDF: (제출 전 추가)
- 발표 자료: (제출 전 추가)
