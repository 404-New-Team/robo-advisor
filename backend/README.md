# Robo-Advisor Backend

FastAPI 기반 백엔드 서버 (포트 8000)  
MySQL DB + AI 서비스(포트 8001) 연동

---

## 아키텍처

```
Streamlit (FE) → Backend :8000 → AI Service :8001
                     ↓
                  MySQL :3306
```

---

## 실행 방법 (Docker)

### 전체 실행 (백엔드 + MySQL)

```bash
docker compose up --build
```

> 처음 실행 시 `--build` 필수. 이후에는 `docker compose up` 만으로 가능.

### 백그라운드 실행

```bash
docker compose up -d --build
```

### 중지

```bash
docker compose down
```

### 재시작

```bash
docker compose restart backend
```

---

## 주요 명령어

### 상태 확인

```bash
# 실행 중인 컨테이너 확인
docker compose ps

# 백엔드 로그 확인
docker compose logs backend

# 실시간 로그 스트리밍
docker compose logs -f backend

# MySQL 로그 확인
docker compose logs mysql
```

### 빌드 관련

```bash
# 코드 변경 후 이미지 재빌드
docker compose up --build

# 캐시 없이 완전 재빌드
docker compose build --no-cache
```

### 컨테이너 접속

```bash
# 백엔드 컨테이너 내부 접속
docker exec -it robo_backend bash

# MySQL 접속
docker exec -it robo_mysql mysql -u robo -probopassword roboadvisor
```

---

## Docker 이미지 및 컨테이너

### 현재 구성

| 컨테이너 | 이미지 | 포트 | 설명 |
|----------|--------|------|------|
| `robo_backend` | `backend-backend` | 8000 | 직접 빌드한 FastAPI 백엔드 |
| `robo_mysql` | `mysql:8.0` | 3306 | Docker Hub 공식 MySQL 이미지 |
| (미구현) | - | 8001 | AI 서비스 — 연결 시 504 오류 정상 |

> AI 서비스(8001)가 없으면 `/optimize`, `/backtest` 등 AI 연동 API는 504를 반환합니다.  
> `/health`, `/docs` 등 백엔드 자체 기능은 정상 동작합니다.

### 이미지 확인

```bash
# 로컬에 있는 Docker 이미지 목록
docker images

# 실행 중인 컨테이너 상태
docker compose ps
```

### 이미지 관리

```bash
# 사용하지 않는 이미지/컨테이너 정리
docker system prune

# 볼륨 포함 전체 정리 (MySQL 데이터도 삭제됨, 주의)
docker compose down -v
```

---

## 포트 충돌 해결

실행 시 포트가 이미 사용 중이라는 오류가 나면:

```bash
# 점유 프로세스 확인
sudo lsof -i :8000
sudo lsof -i :3306

# 좀비 프로세스 종료 (PID는 위 명령어 결과 참고)
sudo kill -9 <PID>
```

---

## API 확인

서버 실행 후 브라우저에서 접속:

| URL | 설명 |
|-----|------|
| `http://localhost:8000/docs` | Swagger UI (API 테스트) |
| `http://localhost:8000/redoc` | ReDoc 문서 |
| `http://localhost:8000/health` | 서버 상태 확인 |

---

## 환경 변수

`.env.example`을 복사해 `.env`로 사용:

```bash
cp .env.example .env
```

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MYSQL_HOST` | `mysql` | MySQL 호스트 (Docker 내부망) |
| `MYSQL_PORT` | `3306` | MySQL 포트 |
| `MYSQL_USER` | `robo` | MySQL 사용자 |
| `MYSQL_PASSWORD` | `robopassword` | MySQL 비밀번호 |
| `MYSQL_DB` | `roboadvisor` | 데이터베이스 이름 |
| `AI_SERVICE_URL` | `http://ai:8001` | AI 서비스 주소 |

---

## 테스트 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 전체 테스트 (MySQL/AI 서비스 불필요)
pytest

# 상세 출력
pytest -v
```
