# Robo-Advisor

에이전틱 RAG와 강화학습 기반 로보어드바이저 프로젝트입니다. 현재 배포 단위는 `ai`, `backend`, `frontend`를 분리하는 방향으로 구성되어 있으며, AWS에서는 각 폴더의 Dockerfile을 기준으로 개별 서버/서비스에 올리는 것을 권장합니다.

## Docker 구조

```text
robo-advisor/
├── ai/
│   ├── Dockerfile              # AI 학습, 실험, AI API 서비스
│   └── docker-compose.yml      # AI 단독 로컬 실행용
├── backend/
│   └── Dockerfile              # FastAPI backend
├── frontend/
│   └── Dockerfile              # Streamlit frontend
└── docker-compose.yml          # 로컬 통합 실행용
```

## 로컬 통합 실행

루트 compose는 로컬에서 전체 연결을 확인하기 위한 용도입니다.

```powershell
docker compose up --build
```

서비스 포트:

| 서비스 | URL | 설명 |
| --- | --- | --- |
| Backend | http://localhost:8000 | FastAPI gateway |
| Backend Swagger | http://localhost:8000/docs | API 문서 |
| AI service | http://localhost:8002 | AI FastAPI service |
| AI service health | http://localhost:8002/health | 모델/SHAP 상태 |
| Frontend | http://localhost:8501 | Streamlit |
| ChromaDB | http://localhost:8001 | Vector DB |
| MySQL | localhost:3306 | Backend DB |

## AI 단독 실행

AI 모듈만 테스트하거나 실험을 돌릴 때는 `ai/` 디렉터리의 compose를 사용합니다.

```powershell
cd ai
docker compose build ai
docker compose run --rm ai
docker compose --profile serve up serve
docker compose --profile train run --rm train
docker compose --profile experiments run --rm shap
```

자세한 AI 실험 실행법과 결과 해석은 [ai/README.md](ai/README.md)를 참고하세요.

## AWS 배포 방향

AWS에서는 루트 compose를 그대로 운영 배포 단위로 쓰기보다, 서비스별 이미지를 따로 빌드하고 배포하는 구성이 적합합니다.

- `ai`: PPO/SHAP/RAG 의존성이 포함된 무거운 AI 서비스
- `backend`: 외부 API gateway 및 DB 기록 담당
- `frontend`: Streamlit 대시보드
- `mysql`, `chromadb`: 운영 환경에서는 RDS, 별도 Chroma 서버, EBS/EFS 볼륨 등으로 분리 가능

## 면책

본 프로젝트는 교육 및 연구 목적입니다. 산출된 포트폴리오, 백테스트 결과, 리서치 응답은 실제 투자 조언이 아니며 미래 수익을 보장하지 않습니다.
