
#!/bin/bash
source $USER/miniconda3/bin/activate openl2s
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# user input test_flg, default false
DATASET_NAME=${1:-mtgjamendo}
TEST_FLG=${2:-false}
START_FROM=${3:-0}
NNODES=${4:-1} # this script only runs on 1 node, you can set to >1, it will increase TOTAL_SHARD
GPU_PER_NODE=${5:-8}
SHARD_PER_GPU=${6:-2}
start_time=$(date +%s)

# PREFIX_TO_REPLACE: 
# save_path_vocal = mixture_path.replace(prefix_to_replace, save_path).replace('.mp3', '.Vocals.mp3')
# save_path_inst = mixture_path.replace(prefix_to_replace, save_path).replace('.mp3', '.Instrumental.mp3')
if [ $DATASET_NAME = harmonix ]; then
# DATA_PATH can be replaced with a filelist.txt
    DATA_PATH=/path/to/harmonix/audio/
    SAVE_PATH=/path/to/harmonix/sep/
    EXP_NAME=harmonix_sep
    PREFIX_TO_REPLACE=/path/to/harmonix/audio/
elif [ $DATASET_NAME = mtgjamendo ]; then
    # DATA_PATH can be replaced with a filelist.txt
    DATA_PATH=/path/to/mtgjamendo/filelist.txt
    SAVE_PATH=/path/to/mtgjamendo/sep/
    EXP_NAME=mtgjamendo_sep
    PREFIX_TO_REPLACE=/path/to/mtgjamendo/audio/
else
    echo "invalid dataset name: $DATASET_NAME"
    exit
fi


mkdir -p $SAVE_PATH
# export OMP_NUM_THREADS=2

TOTAL_SHARD=$((GPU_PER_NODE*SHARD_PER_GPU*NNODES))
echo "total shard: $TOTAL_SHARD, nnodes: $NNODES, gpu_per_node: $GPU_PER_NODE, shard_per_gpu: $SHARD_PER_GPU"
echo "start_from shard: $START_FROM"
echo "data_path: $DATA_PATH"
echo "save_path: $SAVE_PATH"
echo "exp_name: $EXP_NAME"

if [ $TEST_FLG = true ]; then
    echo "testing..."
    echo "overwritting total_shard = 1 and start from shard = 0"
    python main.py \
            --input $DATA_PATH \
            --output $SAVE_PATH \
            --prefix_to_replace $PREFIX_TO_REPLACE \
            --total_shard 1 \
            --cur_shard 0 \
            --resume
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "All conversions completed. Execution time: $execution_time seconds"
    exit
fi

echo "extracting discrete features from raw audio..."

CUR_SHARD=$START_FROM
echo "total shard: $TOTAL_SHARD, start from shard: $START_FROM"
for CUR_GPU in $(seq 0 $((GPU_PER_NODE-1))); do
    echo "gpu $CUR_GPU"
    for SHARD in $(seq 0 $((SHARD_PER_GPU-1))); do
        echo "shard $CUR_SHARD"
        LOG_NAME=$SAVE_PATH/${EXP_NAME}_${CUR_SHARD}_of_${TOTAL_SHARD}.log
        echo "log_file: $LOG_NAME"
        nohup python main.py \
            --input $DATA_PATH \
            --output $SAVE_PATH \
            --prefix_to_replace $PREFIX_TO_REPLACE \
            --total_shard $TOTAL_SHARD \
            --cur_shard $CUR_SHARD \
            --resume \
            --mode 0 \
            --shuffle \
            --cuda_idx $CUR_GPU >${LOG_NAME} 2>&1 &
        
        CUR_SHARD=$((CUR_SHARD+1))
    done
done
wait
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "All conversions completed. Execution time: $execution_time seconds"