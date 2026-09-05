#!/usr/bin/env bash
# Развёртывание бота на чистой Ubuntu. Запускать от root:
#   bash deploy.sh
#
# Скрипт идемпотентный: повторный запуск ничего не ломает, только обновляет.

set -euo pipefail

APP_DIR=/opt/video-translator
SERVICE=vot-bot
CONTAINER=tgbotapi
TG_DATA_DIR=/var/lib/telegram-bot-api

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Запускай от root."

# ── 1. Системные пакеты ─────────────────────────────────────────────────────
say "Ставлю системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  python3 python3-venv python3-pip \
  ffmpeg curl ca-certificates gnupg lsb-release \
  >/dev/null

# ── 2. Docker (для локального Bot API server) ───────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  say "Ставлю Docker"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io >/dev/null
  systemctl enable --now docker
else
  say "Docker уже стоит"
fi

# ── 3. Код и виртуальное окружение ──────────────────────────────────────────
say "Раскладываю код в $APP_DIR"
mkdir -p "$APP_DIR"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SRC_DIR" != "$APP_DIR" ]]; then
  cp -f "$SRC_DIR"/*.py "$APP_DIR"/
  cp -f "$SRC_DIR"/requirements.txt "$APP_DIR"/
  cp -n "$SRC_DIR"/glossary.txt "$APP_DIR"/ 2>/dev/null || true
  cp -n "$SRC_DIR"/.env.example "$APP_DIR"/ 2>/dev/null || true
fi

# Юнит ставим всегда: код мог быть склонирован прямо в APP_DIR
cp -f "$SRC_DIR"/vot-bot.service /etc/systemd/system/"$SERVICE".service

[[ -f "$APP_DIR/.env" ]] || die "Нет $APP_DIR/.env — скопируй .env.example и заполни."

say "Собираю venv и ставлю зависимости"
[[ -d "$APP_DIR/venv" ]] || python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install -q --upgrade pip
"$APP_DIR/venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

# ── 4. Локальный Bot API server ─────────────────────────────────────────────
set +u
source <(grep -E '^(TELEGRAM_API_ID|TELEGRAM_API_HASH)=' "$APP_DIR/.env" || true)
set -u

if [[ -n "${TELEGRAM_API_ID:-}" && -n "${TELEGRAM_API_HASH:-}" ]]; then
  say "Поднимаю локальный Bot API server"
  # Важно: каталог монтируется по тому же пути, что внутри контейнера.
  # В режиме --local getFile отдаёт абсолютный путь внутри контейнера, и бот
  # читает файл с диска напрямую — при именованном volume путь не совпал бы.
  mkdir -p "$TG_DATA_DIR"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d \
    --name "$CONTAINER" \
    --restart always \
    -p 127.0.0.1:8081:8081 \
    -v "$TG_DATA_DIR":/var/lib/telegram-bot-api \
    -e TELEGRAM_API_ID="$TELEGRAM_API_ID" \
    -e TELEGRAM_API_HASH="$TELEGRAM_API_HASH" \
    -e TELEGRAM_LOCAL=1 \
    aiogram/telegram-bot-api:latest >/dev/null
  sleep 3
  docker ps --filter "name=$CONTAINER" --format '  {{.Names}} — {{.Status}}'
else
  say "TELEGRAM_API_ID/TELEGRAM_API_HASH не заданы — Bot API server пропускаю"
  echo "  Без него Telegram режет отдачу файлов на 50 МБ."
  echo "  Ключи берутся на https://my.telegram.org → API development tools."
fi

# ── 5. Ночная уборка кэша Bot API server ────────────────────────────────────
say "Ставлю ночную очистку кэша"
cat > /etc/cron.d/tgbotapi-cleanup <<CRON
# Чистим кэш локального Bot API server от файлов старше суток
17 4 * * * root find $TG_DATA_DIR -type f -mtime +1 -delete >/dev/null 2>&1
CRON
chmod 644 /etc/cron.d/tgbotapi-cleanup

# ── 6. systemd ──────────────────────────────────────────────────────────────
say "Запускаю сервис $SERVICE"
systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null
systemctl restart "$SERVICE"
sleep 3
systemctl --no-pager --lines=15 status "$SERVICE" || true

say "Готово. Логи: journalctl -u $SERVICE -f"
