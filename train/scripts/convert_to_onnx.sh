#!/bin/bash

# MODIFY PATH AND PARTITION BEFORE USE
export ROOT=xxx
export PLUGINPATH=xxx
export PARTITION=ModelTC

T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2

CPUS_PER_TASK=${CPUS_PER_TASK:-4}
export PYTHONPATH=$ROOT:$PYTHONPATH
export EXCLUDE_TASKS=det_3d:action

srun -p $PARTITION -n8 --ntasks-per-node=8 --gres=gpu:8 \
  python -m up to_onnx \
  --config=$cfg \
  --save_prefix=toonnx_256 \
  --input_size=3x256x256 \
  2>&1 | tee log.toonnx.$T.txt
