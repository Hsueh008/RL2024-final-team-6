export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


accelerate launch \
    --config_file train/config/acc_config.yaml \
    --num_cpu_threads_per_process 6 \
    train/train_dpo.py \
