set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

project := "robo-advisor"
compose_file := "docker-compose.prod.yml"
branch := "develop"
frontend_url := "https://robby.elhye.com/"
health_url := "https://robby.elhye.com/api/health"

# Show available commands
_default:
    just --list

# Deploy latest origin/develop and rebuild only services affected by changed files
deploy:
    @test -f .env || { echo "missing .env" >&2; exit 1; }
    @if [ -n "$(git status --porcelain)" ]; then \
        echo "Refusing to deploy with uncommitted changes:" >&2; \
        git status --short >&2; \
        exit 1; \
    fi
    @before="$(git rev-parse HEAD)"; \
    git fetch origin {{branch}}; \
    git checkout {{branch}}; \
    git reset --hard origin/{{branch}}; \
    test -f {{compose_file}} || { echo "missing {{compose_file}} after update" >&2; exit 1; }; \
    after="$(git rev-parse HEAD)"; \
    if [ "$before" = "$after" ]; then changed=""; else changed="$(git diff --name-only "$before" "$after")"; fi; \
    echo "Changed files:"; \
    if [ -n "$changed" ]; then echo "$changed"; else echo "  none"; fi; \
    services=""; \
    need_backend=0; need_frontend=0; need_ai=0; \
    if ! docker image inspect robo-advisor-backend:prod >/dev/null 2>&1; then need_backend=1; fi; \
    if ! docker image inspect robo-advisor-frontend:prod >/dev/null 2>&1; then need_frontend=1; fi; \
    if ! docker image inspect robo-advisor-ai:prod >/dev/null 2>&1; then need_ai=1; fi; \
    while IFS= read -r file; do \
        [ -n "$file" ] || continue; \
        case "$file" in \
            backend/*) need_backend=1 ;; \
            frontend/*) need_frontend=1 ;; \
            ai/*) need_ai=1 ;; \
            docker-compose.prod.yml) need_backend=1; need_frontend=1; need_ai=1 ;; \
        esac; \
    done < <(printf '%s\n' "$changed"); \
    if [ "$need_backend" = "1" ]; then services="$services backend"; fi; \
    if [ "$need_frontend" = "1" ]; then services="$services frontend"; fi; \
    if [ "$need_ai" = "1" ]; then services="$services ai"; fi; \
    if [ -n "$services" ]; then \
        echo "Building changed services:$services"; \
        docker compose -p {{project}} -f {{compose_file}} build $services; \
    else \
        echo "No service image changes detected; skipping build"; \
    fi; \
    docker compose -p {{project}} -f {{compose_file}} up -d; \
    just wait-ready; \
    curl -fsS --max-time 15 {{health_url}} >/dev/null; \
    curl -fsSI --max-time 15 {{frontend_url}} >/dev/null; \
    echo "deployed origin/{{branch}} ($after)"

# Force rebuild all application images, then deploy
deploy-full:
    git fetch origin {{branch}}
    git checkout {{branch}}
    git reset --hard origin/{{branch}}
    docker compose -p {{project}} -f {{compose_file}} up -d --build
    just wait-ready
    curl -fsS --max-time 15 {{health_url}} >/dev/null
    curl -fsSI --max-time 15 {{frontend_url}} >/dev/null
    @echo "deployed origin/{{branch}} with full rebuild"

# Wait until public frontend and backend health endpoints are ready
wait-ready:
    @printf 'waiting for API health'; \
    for i in $(seq 1 60); do \
        if curl -fsS --max-time 5 {{health_url}} >/dev/null; then echo ' ready'; break; fi; \
        printf '.'; sleep 2; \
        if [ "$i" = "60" ]; then echo ' timeout' >&2; exit 1; fi; \
    done
    @printf 'waiting for frontend'; \
    for i in $(seq 1 60); do \
        if curl -fsSI --max-time 5 {{frontend_url}} >/dev/null; then echo ' ready'; break; fi; \
        printf '.'; sleep 2; \
        if [ "$i" = "60" ]; then echo ' timeout' >&2; exit 1; fi; \
    done

# Restart production services without changing git state
restart:
    docker compose -p {{project}} -f {{compose_file}} up -d
    just wait-ready
    docker compose -p {{project}} -f {{compose_file}} ps

# Follow logs for a service, e.g. just logs ai
logs service="backend":
    docker compose -p {{project}} -f {{compose_file}} logs -f {{service}}

# Show production service status
ps:
    docker compose -p {{project}} -f {{compose_file}} ps
