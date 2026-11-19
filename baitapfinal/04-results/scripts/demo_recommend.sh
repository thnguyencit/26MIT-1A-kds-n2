#!/usr/bin/env bash
# Ví dụ gợi ý size bằng mô hình đã huấn luyện (cần chạy bước 4 của run_all.sh trước).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
M=../04-results/models
LOG=../04-results/logs/demo_recommend.txt
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COLUMNS=120
: > "$LOG"
demo() { echo "\$ sizerec $*" | tee -a "$LOG"; echo | tee -a "$LOG"; "$PY" -m sizerec "$@" 2>&1 | tee -a "$LOG"; echo | tee -a "$LOG"; }

# RentTheRunway, khách mới (cold-start): chỉ có số đo tự khai
demo recommend $M/rtr.pt --item 126335 --height 165cm --weight 57kg --bust-size 34b --body-type petite --age 22 --explain
# RentTheRunway, khách cũ có nhiều lịch sử
demo recommend $M/rtr.pt --item 126335 --user 691468 --explain
# ModCloth, khách mới
demo recommend $M/modcloth.pt --item 539980 --height 5ft6in --bra-size 34 --cup b --hips 38 --explain
# Đầu ra JSON (để tích hợp vào hệ thống khác)
demo recommend $M/rtr.pt --item 126335 --height 168cm --weight 59kg --json
