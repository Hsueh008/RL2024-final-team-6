# RL2024-final-team-6

## Build Training Environment

1. Using `micromamba` to maintain the environment.

```bash
micromamba create -n RL2024-final-env python=3.10
micromamba activate RL2024-final-env
```

2. Install modules.

```bash
pip install -r requirements.txt
micromamba install -c nvidia cuda-toolkit=12.6 # <-- It's better to check wich version of your CUDA.
micromamba install -c conda-forge ninja
```

3. Compile `flash-attn` from github

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation 
```

4. Setup the `accelerate.yaml`, and follow the settings.

```bash
accelerate config
```
```bash
---------------------------------------------
In which compute environment are you running?
This machine   
---------------------------------------------
Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: # Cluster Computing, if need it, otherwise input 1 or just [ENTER].
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: # [ENTER]
Do you wish to optimize your script with torch dynamo?[yes/NO]: # [ENTER]
Do you want to use DeepSpeed? [yes/NO]: # yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: # yes
Please enter the path to the json DeepSpeed config file: # path to train/config/ds_config.json
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: # [ENTER]
Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]: # [ENTER]
How many GPU(s) should be used for distributed training? [1]: # According to you computing resources.
accelerate configuration saved at .../train/config/ds_config.json
```

## Training

```bash
bash train/train_dpo.py
```

> Change the hyperparameters in `train/train_dpo.py`.

## Evaluation

1. Clone VerilogEval

```bash
git clone https://github.com/NVlabs/verilog-eval.git
```

2. Install ICARUS Verilog

```bash
git clone https://github.com/steveicarus/iverilog.git && cd iverilog && git checkout v12-branch
PREFIX=$CONDA_PREFIX # <-- Install to your environment.
./configure --prefix=$PREFIX
make -j4 # <-- According to how many CPUs you want to use.
make install
```

3. Evaluate

```bash
bash eval/eval.sh
```