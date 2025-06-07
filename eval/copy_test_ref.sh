#!/usr/bin/env bash
set -e
shopt -s nullglob   # 如果 glob 沒配到，回傳空陣列而不是字串本身

# --------- 預設路徑，可用 -s -d 覆蓋 ----------
SRC_DIR="dataset_spec-to-rtl"
DST_DIR="build_generation"

while getopts "s:d:h" opt; do
  case "$opt" in
    s) SRC_DIR="$OPTARG" ;;
    d) DST_DIR="$OPTARG" ;;
    h)
      echo "用法: $0 [-s SRC_DIR] [-d DST_DIR]"
      exit 0
      ;;
    *) exit 1 ;;
  esac
done

# --------- 開始複製 ----------
for prob_dir in "${DST_DIR}"/*/; do
  prob_name=$(basename "${prob_dir%/}")

  prompt_src="${SRC_DIR}/${prob_name}_prompt.txt"
  test_src="${SRC_DIR}/${prob_name}_test.sv"
  ref_src="${SRC_DIR}/${prob_name}_ref.sv"

  copied=false

  # ---- prompt ----
  if [[ -f "$prompt_src" ]]; then
    cp -u "$prompt_src" "${prob_dir}"
    echo "✅  copied prompt --> ${prob_name}/${prob_name}_prompt.txt"
    copied=true
  else
    echo "⚠️  missing prompt: ${prob_name}_prompt.txt"
  fi

  # ---- test ----
  if [[ -f "$test_src" ]]; then
    cp -u "$test_src" "${prob_dir}"
    echo "✅  copied test   --> ${prob_name}/${prob_name}_test.sv"
    copied=true
  else
    echo "⚠️  missing test  : ${prob_name}_test.sv"
  fi

  # ---- ref ----
  if [[ -f "$ref_src" ]]; then
    cp -u "$ref_src" "${prob_dir}"
    echo "✅  copied ref    --> ${prob_name}/${prob_name}_ref.sv"
    copied=true
  else
    echo "⚠️  missing ref   : ${prob_name}_ref.sv"
  fi

  if ! $copied; then
    echo "❌  ${prob_name}: 沒有任何 prompt/test/ref 被複製"
  fi
done
