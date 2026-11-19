#!/usr/bin/env bash
# Chạy lại TOÀN BỘ thực nghiệm và tái tạo thư mục 04-results.
#   bash scripts/run_all.sh                 # đầy đủ (~20-30 phút trên CPU 4 luồng)
#   SKIP_EXPERIMENT=1 bash scripts/run_all.sh   # bỏ bước 3 (dùng lại kết quả thực nghiệm đã có)
#   PYTHON=python3 bash scripts/run_all.sh  # chọn trình thông dịch Python
set -euo pipefail
cd "$(dirname "$0")/.."                      # -> 03-code
PY=${PYTHON:-python}
DATA=../02-data
RES=../04-results
MC=$DATA/modcloth_final_data.json.gz
RTR=$DATA/renttherunway_final_data.json.gz
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COLUMNS=120
mkdir -p "$RES/logs" "$RES/models" "$RES/experiment"

run() {  # run <file log> <lệnh sizerec ...>: in ra màn hình và ghi log
  local log="$RES/logs/$1"; shift
  echo "\$ sizerec $*" | tee "$log"; echo | tee -a "$log"
  "$PY" -m sizerec "$@" 2>&1 | tee -a "$log"
}

echo "== 1. Môi trường và thống kê dữ liệu"
run info.txt info
run data_modcloth.txt data "$MC"
run data_rtr.txt data "$RTR"

echo "== 2. Kiểm thử pipeline trên dữ liệu giả lập"
"$PY" tests/test_smoke.py 2>&1 | tee "$RES/logs/smoke_test.txt"

if [ "${SKIP_EXPERIMENT:-0}" != "1" ]; then
  echo "== 3. Thực nghiệm so sánh 7 mô hình, 2 giao thức, 10 seed, kiểm định thống kê"
  run experiment_modcloth.txt experiment "$MC" --protocols cold warm --seeds 1 2 3 4 5 6 7 8 9 10 --boot 200 -o "$RES/experiment/modcloth"
  run experiment_rtr.txt experiment "$RTR" --protocols cold warm --seeds 1 2 3 4 5 6 7 8 9 10 --boot 200 -o "$RES/experiment/rtr"
fi

echo "== 4. Huấn luyện mô hình đề xuất trên toàn bộ dữ liệu (dùng cho gợi ý size)"
run train_modcloth.txt train "$MC" -o "$RES/models/modcloth.pt"
run train_rtr.txt train "$RTR" -o "$RES/models/rtr.pt"

echo "== 5. Ví dụ gợi ý size"
PYTHON="$PY" bash scripts/demo_recommend.sh

echo "== 6. Vẽ hình và xuất Excel"
"$PY" scripts/make_figures.py "$RES" | tee "$RES/logs/make_figures.txt"
"$PY" scripts/make_figures_en.py "$RES" | tee -a "$RES/logs/make_figures.txt"
"$PY" scripts/export_excel.py "$RES" | tee "$RES/logs/export_excel.txt"
echo "Xong. Kết quả trong $RES"
