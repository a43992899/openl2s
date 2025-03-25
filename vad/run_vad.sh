#!/bin/bash

IN=$1       # filelist_or_dir of the address of the music separation
LOG_ROOT=$2
JOBS=$3

script_path=$(readlink -f "$0")
cd $(dirname "$(dirname "$script_path")")

export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1

python vad/vad_webrtcvad.py \
        --filelist_or_dir $IN \
        --out_dir $LOG_ROOT \
        --jobs $JOBS