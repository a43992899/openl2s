from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FireRedTeam/FireRedASR-AED-L",
    local_dir="./pretrained_models",
    local_dir_use_symlinks=False,
    repo_type="model",
)
