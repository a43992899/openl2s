# openl2s

# Download [Muchin](https://github.com/CarlWangChina/MuChin) (Academic Only!!!)
Song audio cannot be used for commercial model training without the authorization of the copyright holder. Audio download link: https://pan.baidu.com/s/1D4xGQhYUwWbpaHyAS71dfw?pwd=1234 Extract password: 1234

# Prepare Source Separation Pipeline

```bash
conda create -n openl2s python=3.10
conda activate openl2s
cd sep
conda install -c conda-forge libsndfile pysoundfile
conda install -c conda-forge ffmpeg
pip install torch==2.1.0 torchaudio
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