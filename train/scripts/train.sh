#!/bin/bash

# MODIFY PATH AND PARTITION BEFORE USE
export ROOT=xxx
export PLUGINPATH=xxx
export PARTITION=ModelTC

T=$(date +%m%d%H%M)
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH

export EXCLUDE_TASKS=det_3d:action


srun -p $PARTITION -n8 --ntasks-per-node=8 --gres=gpu:8 \
python -m up train \
  --config=$cfg \
  --display=40 \
  2>&1 | tee log.train.$T.txt
