export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --num_cpu_threads_per_process 6 \
    code/DPO/train.py \
