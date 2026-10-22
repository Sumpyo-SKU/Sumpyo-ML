#!/usr/bin/env bash
# EC2 cron에서 10분마다 실행: .env의 DB 값을 읽어서 predict.py 실행
set -euo pipefail
cd "$(dirname "$0")"
ENV_FILE="${ENV_FILE:-$HOME/sumpyo/.env}"     # 스프링이 쓰는 .env 경로로 맞추기
set -a; source "$ENV_FILE"; set +a
export DB_HOST=127.0.0.1 DB_PORT=5432 DB_NAME=sumpyo
export DB_USER="${DB_USERNAME:-sumpyo}"
exec ./.venv/bin/python predict.py "$@"
 