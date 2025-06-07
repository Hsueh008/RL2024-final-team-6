#!/bin/bash

model="model/deepseek-coder-6.7b-dpo-lora"
build_dir="verilog-eval/build_generation"
budgets=20 # n=20 <-- 根據VerilogEval論文
task="spec-to-rtl"
few_shot=1

mkdir -p ${build_dir}

# 生成verilog程式碼
python eval/generate_sample.py \
    --model $model \
    --budgets $budgets \
    --build_dir $build_dir \
    --few_shot $few_shot

# 複製_test.sv和ref.sv參考資料
bash eval/copy_test_ref.sh -d ${build_dir} -s verilog-eval/dataset_${task}

cd ${build_dir}
../configure \
    --with-model=$model \
    --with-examples=1 \
    --with-samples=$budgets \
    --with-task=spec-to-rtl

cd ../..
echo ${PWD}
python eval/analyze.py --end ${budgets}