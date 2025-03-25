# openl2s

# Prepare Source Separation Pipeline

```bash
conda create -n openl2s python=3.10
conda activate openl2s
cd sep
pip install -r requirements.txt

# login to huggingface cli
huggingface-cli login

# download ckpt from hf
python download_sep_ckpt.py
```

This pipeline uses 3 models ensemble (minimal signal) to separate the source.

Example usage:

```bash
# single node
# for more details about the args, see sep/run.multigpu.sh
cd sep
bash run.multigpu.sh mtgjamendo false 0 1 8 2

# slurm multinode
sbatch run.slurm.seperate.sh
```

# Prepare VAD and ASR Pipline

```bash
conda activate openl2s

cd asr

# login to huggingface cli
huggingface-cli login

# download ckpt from hf
python download_asr_ckpt.py

```
Example usage:

```bash
# single node
# for more details about the args, see run_vad_asr.sh
bash run_vad_asr.sh

```