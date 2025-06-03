from modelscope import snapshot_download

def download_model(model_name: str, local_dir: str = "./"):
    snapshot_download(repo_id=model_name, local_dir=local_dir)

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    local_dir = "./"
    download_model(model_name, local_dir)
    print(f"Model {model_name} downloaded to {local_dir}")
