# RUNBOOK — Robo-Advisor 실행 가이드

## 목차
1. [사전 요구사항](#사전-요구사항)
2. [환경 변수 설정](#환경-변수-설정)
3. [Docker Compose로 전체 실행 (권장)](#docker-compose로-전체-실행-권장)
4. [로컬에서 서비스별 개별 실행](#로컬에서-서비스별-개별-실행)
5. [DB 마이그레이션](#db-마이그레이션)
6. [포트 구성 요약](#포트-구성-요약)
7. [상태 확인](#상태-확인)
8. [종료](#종료)

---

## 사전 요구사항

| 항목 | 버전 |
|---|---|
| Docker | 24.0 이상 |
| Docker Compose | v2.0 이상 |
| Python | 3.10 / 3.11 (로컬 실행 시) |
| Git | 2.x 이상 |

---

## 환경 변수 설정

프로젝트 루트의 `.env.example`을 복사해 `.env`를 생성한다.

```bash
cp .env.example .env
```

`.env` 필수 항목을 채운다.

```dotenv
# Anthropic Claude API 키 (Research Agent LLM 호출에 필요)
ANTHROPIC_API_KEY=sk-ant-...

# JWT 서명 키 (임의의 긴 문자열로 교체 권장)
JWT_SECRET_KEY=change-me-in-production

# 나머지 항목은 Docker Compose 기본값으로 동작
```

> `ANTHROPIC_API_KEY`가 없으면 `/research` 엔드포인트가 LLM 없이 추출형 리포트로 폴백된다.

---

## Docker Compose로 전체 실행 (권장)

### 1. 이미지 빌드 및 전체 서비스 시작

```bash
docker compose up --build
```

최초 실행 시 이미지 빌드에 수 분이 소요된다. 이후 실행은 캐시를 활용해 빠르다.

### 2. 백그라운드 실행

```bash
docker compose up -d --build
```

### 3. 서비스 시작 순서

Docker Compose가 아래 순서로 의존성을 처리한다.

```
MySQL (healthy) ─┐
                  ├─► Backend ─► Frontend
AI 서비스 ────────┘
    │
ChromaDB
```

MySQL `healthcheck` 통과 후 Backend가 기동되므로, 처음에는 30~60초 대기가 필요할 수 있다.

### 4. 접속 확인

| 서비스 | URL |
|---|---|
| Streamlit 대시보드 | http://localhost:8501 |
| Backend API 문서 (Swagger) | http://localhost:8080/docs |
| AI 서비스 API 문서 (Swagger) | http://localhost:8002/docs |
| ChromaDB | http://localhost:8001 |
| MySQL | `localhost:3307` (DBeaver로 접속) |

---

## 로컬에서 서비스별 개별 실행

Docker 없이 각 서비스를 직접 실행한다. MySQL과 ChromaDB는 별도로 실행되어 있어야 한다.

### 공통 준비

```bash
# MySQL, ChromaDB는 Docker로만 띄우고 나머지를 로컬 실행하는 혼합 방식 권장
docker compose up -d mysql chromadb
```

---

### AI 서비스

```bash
cd ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

- Swagger: http://localhost:8001/docs
- PPO 체크포인트가 없으면 자동으로 MVO 폴백으로 동작한다.

---

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 환경 변수 설정 (로컬 실행 시 오버라이드)
export MYSQL_HOST=localhost
export MYSQL_PORT=3307
export AI_SERVICE_URL=http://localhost:8001

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger: http://localhost:8000/docs

---

### Frontend

```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export API_BASE_URL=http://localhost:8000

streamlit run streamlit_app/app.py --server.port 8501
```

- 대시보드: http://localhost:8501

---

### Mock 모드 (백엔드 없이 프론트만 실행)

```bash
export ROBBY_USE_MOCK=true
streamlit run streamlit_app/app.py
```

백엔드·AI 서비스 없이 더미 데이터로 UI를 확인할 수 있다.

---

## DB 마이그레이션

MySQL이 기동된 상태에서 Backend 컨테이너(또는 로컬) 내부에서 실행한다.

### Docker Compose 환경

```bash
docker compose exec backend alembic upgrade head
```

### 로컬 환경

```bash
cd backend
export MYSQL_HOST=localhost
export MYSQL_PORT=3307

alembic upgrade head
```

### 마이그레이션 이력 확인

```bash
alembic history --verbose
alembic current
```

---

## 포트 구성 요약

| 서비스 | 호스트 포트 | 컨테이너 내부 포트 |
|---|---|---|
| Streamlit (Frontend) | **8501** | 8501 |
| Backend (FastAPI) | **8080** | 8000 |
| AI 서비스 (FastAPI) | **8002** | 8001 |
| ChromaDB | **8001** | 8000 |
| MySQL | **3307** | 3306 |

> 로컬에서 직접 실행 시 Backend는 `:8000`, AI 서비스는 `:8001`, MySQL은 `:3306`을 사용한다.

---

## 상태 확인

### 전체 컨테이너 상태

```bash
docker compose ps
```

### Backend 헬스체크

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "models": { "rl_engine": true, "rag_agent": true }
}
```

### AI 서비스 헬스체크

```bash
curl http://localhost:8002/health
```

```json
{
  "status": "ok",
  "rl_engine": true,
  "rag_agent": true,
  "shap_explainer": true
}
```

### 로그 확인

```bash
# 전체 서비스
docker compose logs -f

# 특정 서비스
docker compose logs -f backend
docker compose logs -f ai
docker compose logs -f frontend
```

---

## 종료

```bash
# 컨테이너 중지 (볼륨 유지)
docker compose down

# 컨테이너 + 볼륨 모두 삭제 (DB 초기화 포함)
docker compose down -v
```
