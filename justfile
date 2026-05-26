set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

project := "robo-advisor"
compose_file := "docker-compose.prod.yml"
branch := "develop"
frontend_url := "https://robby.elhye.com/"
health_url := "https://robby.elhye.com/api/health"

# Show available commands
_default:
    just --list

# Deploy the latest origin/develop with production Docker Compose
deploy:
    @test -f {{compose_file}} || { echo "missing {{compose_file}}" >&2; exit 1; }
    @test -f .env || { echo "missing .env" >&2; exit 1; }
    @if [ -n "$(git status --porcelain)" ]; then \
        echo "Refusing to deploy with uncommitted changes:" >&2; \
        git status --short >&2; \
        exit 1; \
    fi
    git fetch origin {{branch}}
    git checkout {{branch}}
    git reset --hard origin/{{branch}}
    docker compose -p {{project}} -f {{compose_file}} up -d --build
    docker compose -p {{project}} -f {{compose_file}} ps
    curl -fsS --max-time 15 {{health_url}} >/dev/null
    curl -fsSI --max-time 15 {{frontend_url}} >/dev/null
    @echo "deployed origin/{{branch}}"

# Restart production services without changing git state
restart:
    docker compose -p {{project}} -f {{compose_file}} up -d
    docker compose -p {{project}} -f {{compose_file}} ps

# Follow logs for a service, e.g. just logs ai
logs service="backend":
    docker compose -p {{project}} -f {{compose_file}} logs -f {{service}}

# Show production service status
ps:
    docker compose -p {{project}} -f {{compose_file}} ps
