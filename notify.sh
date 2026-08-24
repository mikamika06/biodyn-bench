#!/bin/bash
# Сповіщення про віхи автономного прогону.
# Щоб додати Telegram: впиши свій токен і chat_id нижче.
MSG="$1"
STAMP=$(date "+%Y-%m-%d %H:%M")
LOG="$(dirname "$0")/out/MILESTONES.txt"
mkdir -p "$(dirname "$LOG")"
echo "[$STAMP] $MSG" >> "$LOG"
osascript -e "display notification \"$MSG\" with title \"biodyn-bench\"" 2>/dev/null
printf '\a'
# TG_TOKEN=""; TG_CHAT=""
# [ -n "$TG_TOKEN" ] && curl -s -X POST "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
#   -d chat_id="$TG_CHAT" -d text="biodyn-bench: $MSG" >/dev/null
echo "$MSG"
