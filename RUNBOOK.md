# Robo-Advisor Deployment Runbook

이 문서는 단일 서버에서 Docker Compose로 운영 배포할 때의 최소 절차입니다. 같은 서버에 다른 서비스가 떠 있을 수 있으므로 운영 compose는 내부 서비스 포트를 외부에 열지 않습니다.

## 1. 사전 확인

```bash
docker ps
sudo ss -tulpn | grep LISTEN
```

기본 운영 노출 포트는 다음과 같습니다. 기본 바인드는 `127.0.0.1`이라 외부에는 직접 공개되지 않고, 분리된 `proxy-nginx`가 외부 80/443을 담당합니다.

| 서비스 | 기본 호스트 바인드 | 기본 호스트 포트 | 컨테이너 포트 |
| --- | --- | ---: | ---: |
| frontend | 127.0.0.1 | 18501 | 8501 |
| backend | 127.0.0.1 | 18080 | 8000 |

포트가 겹치면 `.env`에서 `FRONTEND_PORT`, `BACKEND_PORT`를 변경하세요. nginx 컨테이너와 직접 Docker DNS로 연결할 때는 `proxy` external network를 사용합니다.

## 2. 환경변수 설정

```bash
cp .env.example .env
```

운영 배포 전 최소한 아래 값은 바꾸세요.

```dotenv
ANTHROPIC_API_KEY=...
JWT_SECRET_KEY=...
MYSQL_ROOT_PASSWORD=...
MYSQL_PASSWORD=...
BACKEND_BIND=127.0.0.1
BACKEND_PORT=18080
FRONTEND_BIND=127.0.0.1
FRONTEND_PORT=18501
```

`JWT_SECRET_KEY`, `MYSQL_ROOT_PASSWORD`, `MYSQL_PASSWORD`는 기본값으로 운영 배포하지 마세요.

## 3. 배포

```bash
docker compose -p robo-advisor -f docker-compose.prod.yml up -d --build
docker compose -p robo-advisor -f docker-compose.prod.yml ps
```

로컬 확인 URL:

```text
Frontend: http://127.0.0.1:18501
Backend:  http://127.0.0.1:18080
Swagger:  http://127.0.0.1:18080/docs
Health:   http://127.0.0.1:18080/health
```

외부 접속은 `proxy-nginx` 도메인 라우팅을 통해 여세요. 서버 IP로 직접 열어야 하면 `.env`에서 `FRONTEND_BIND=0.0.0.0`, `BACKEND_BIND=0.0.0.0`으로 바꿀 수 있지만, 운영에서는 권장하지 않습니다.

## 4. 로그 확인

```bash
docker compose -p robo-advisor -f docker-compose.prod.yml logs -f backend
docker compose -p robo-advisor -f docker-compose.prod.yml logs -f ai
docker compose -p robo-advisor -f docker-compose.prod.yml logs -f frontend
```

## 5. 업데이트

```bash
git pull
docker compose -p robo-advisor -f docker-compose.prod.yml up -d --build
docker compose -p robo-advisor -f docker-compose.prod.yml ps
```

## 6. 중지

컨테이너만 중지하고 데이터 볼륨은 유지합니다.

```bash
docker compose -p robo-advisor -f docker-compose.prod.yml down
```

데이터 볼륨까지 삭제해야 할 때만 아래 명령을 사용하세요.

```bash
docker compose -p robo-advisor -f docker-compose.prod.yml down -v
```

## 7. Nginx 연결 예시

현재 서버는 분리된 `proxy-nginx` 컨테이너가 80/443을 담당하고, `/home/lickelon/projects/proxy/conf.d`를 `/etc/nginx/conf.d`로 마운트합니다. robo-advisor 운영 compose는 external network `proxy`에 `robo-advisor-frontend`, `robo-advisor-backend` alias로 붙습니다.

예시 파일: `/home/lickelon/projects/proxy/conf.d/robo.example.com.conf`

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name robo.example.com;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    http2 on;
    server_name robo.example.com;

    resolver 127.0.0.11 ipv6=off valid=30s;
    set $robo_frontend http://robo-advisor-frontend:8501;
    set $robo_backend http://robo-advisor-backend:8000;

    ssl_certificate /etc/letsencrypt/live/robo.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/robo.example.com/privkey.pem;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location /api/ {
        rewrite ^/api/(.*)$ /$1 break;
        proxy_pass $robo_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    location / {
        proxy_pass $robo_frontend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }
}
```

적용 전 검증:

```bash
docker exec proxy-nginx nginx -t
docker exec proxy-nginx nginx -s reload
```

인증서가 아직 없다면 먼저 80 only server block과 certbot으로 인증서를 발급한 뒤 443 server block을 추가하세요.
