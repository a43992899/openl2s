from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="m-a-p/openl2s_sep_ckpts",
    local_dir="./openl2s_sep_ckpts",
    local_dir_use_symlinks=False,
    repo_type="model",
)
