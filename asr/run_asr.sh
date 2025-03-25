#!/bin/bash

IN=$1
OUT=$2
JOBS=$3

script_path=$(readlink -f "$0")
cd $(dirname "$(dirname "$script_path")")

export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1

python asr/speech2text_mp.py $IN $OUT $JOBS