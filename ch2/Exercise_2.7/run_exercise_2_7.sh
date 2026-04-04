#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

EPOCHS="${EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-50}"
CLEAN="${CLEAN:-0}"

if [[ "${CLEAN}" == "1" ]]; then
  rm -f MSE_*.mat Figure_2_9_reproduced.png Figure_2_9_reproduced.pdf
  rm -f dnn_ce/*.npz 2>/dev/null || true
fi

# fail fast on syntax errors
python -m py_compile main.py tools/raputil.py tools/networks.py plot_figure.py

MAIN_BAK="$(mktemp)"
cp main.py "${MAIN_BAK}"

restore_main() {
  cp "${MAIN_BAK}" main.py
  rm -f "${MAIN_BAK}"
}
trap restore_main EXIT

set_main_cfg() {
  local CE_TYPE="$1"
  local TEST_CE="$2"
  local CP_FLAG="$3"
  local EPOCHS_LOCAL="$4"
  local BATCH_LOCAL="$5"

  python - "${CE_TYPE}" "${TEST_CE}" "${CP_FLAG}" "${EPOCHS_LOCAL}" "${BATCH_LOCAL}" <<'PY'
from pathlib import Path
import re
import sys

ce_type, test_ce, cp_flag, epochs, batch = sys.argv[1:]

path = Path("main.py")
text = path.read_text()

replacements = [
    (r"(?m)^training_epochs = .*$", f"training_epochs = {epochs}"),
    (r"(?m)^batch_size = .*$", f"batch_size = {batch}"),
    (r"(?m)^ce_type = .*$", f"ce_type = '{ce_type}'  # channel estimation: 'mmse', 'dnn'"),
    (r"(?m)^test_ce = .*$", f"test_ce = {test_ce}"),
    (r"(?m)^CP_flag = .*$", f"CP_flag = {cp_flag}"),
    (r"(?m)^NoCP = \(CP_flag is False\).*$", "NoCP = (CP_flag is False)"),
]

for pattern, repl in replacements:
    text_new, count = re.subn(pattern, repl, text)
    if count != 1:
        raise SystemExit(f"Failed to patch main.py with pattern: {pattern}")
    text = text_new

path.write_text(text)
print(f"[patched] ce_type={ce_type}, test_ce={test_ce}, CP_flag={cp_flag}, epochs={epochs}, batch={batch}")
PY
}

run_case() {
  local NAME="$1"
  local CE_TYPE="$2"
  local TEST_CE="$3"
  local CP_FLAG="$4"

  echo
  echo "=============================="
  echo "Running: ${NAME}"
  echo "=============================="

  set_main_cfg "${CE_TYPE}" "${TEST_CE}" "${CP_FLAG}" "${EPOCHS}" "${BATCH_SIZE}"
  python main.py
}

run_case "DNN train with CP"      "dnn"  "False" "True"
run_case "DNN test with CP"       "dnn"  "True"  "True"
run_case "LMMSE test with CP"     "mmse" "True"  "True"

run_case "DNN train without CP"   "dnn"  "False" "False"
run_case "DNN test without CP"    "dnn"  "True"  "False"
run_case "LMMSE test without CP"  "mmse" "True"  "False"

python plot_figure.py

echo
echo "Done."
echo "Generated files:"
echo "  MSE_dnn_64QAM.mat"
echo "  MSE_mmse_64QAM.mat"
echo "  MSE_dnn_64QAM_CP_FREE.mat"
echo "  MSE_mmse_64QAM_CP_FREE.mat"
echo "  Figure_2_9_reproduced.png"
echo "  Figure_2_9_reproduced.pdf"