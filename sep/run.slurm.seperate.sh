#!/bin/bash
#SBATCH -p audio
#SBATCH -J sep                  # job name
#SBATCH -N 2                    # num_node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8        
#SBATCH --cpus-per-task=220       
#SBATCH --mem=1280G               
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH -t 72:00:00



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
echo SLURM_PROCID: $SLURM_PROCID
echo SLURM_NODEID: $SLURM_NODEID
export LOGLEVEL=INFO


PROJECT_ROOT=/aifs4su/mmcode/codeclm

srun -l --container-image $PROJECT_ROOT/containers/pytorch-23.10-py3-code-cli-torchaudio.sqsh --container-writable --container-remap-root --no-container-mount-home \
    --container-mounts $PROJECT_ROOT/openl2s:/workspace/openl2s,/aifs4su/:/aifs4su/ \
    --container-workdir /workspace/openl2s bash -c "\
     bash run.multigpu.sh mtgjamendo false 0 1 8 6"
