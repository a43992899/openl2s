#!/bin/bash

# source $USER/miniconda3/bin/activate openl2s

stage=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_NAME="mtgjamendo"   # Name of the dataset to be processed
DATA_ROOT=/path/to/$DATA_NAME
LOG_ROOT=/path/to/result/$DATA_NAME
mkdir -p "$LOG_ROOT"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#GPU_ARRAY[@]}

if [ "$stage" -le 0 ]; then

    echo "stage 0: run vad"
    bash vad/run_vad.sh \
        $DATA_ROOT/sep \
        $LOG_ROOT \
        10
fi

if [ "$stage" -le 1 ]; then

    echo "stage 1: run asr"
    python asr/utils/gen_json.py \
        $DATA_ROOT/sep \
        $LOG_ROOT/vad_info \
        $DATA_ROOT/${DATA_NAME}_data.json

    bash asr/run_asr.sh \
        $DATA_ROOT/${DATA_NAME}_data.json \
        $LOG_ROOT/lrc \
        $((GPU_COUNT * 1)) \
        $DATA_NAME
fi
