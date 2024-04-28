from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf

import requests
from huggingface_hub import hf_hub_download

MODEL = "rwkv-4-pile-169m"
DEFAULT_TOKENIZER_URL = "https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B/raw/main/20B_tokenizer.json"

TOKENIZER_PATH = Path(__file__).parent / "20B_tokenizer.json"
models = {
    "raven-14b-ctx4096": {
        "repo_id": "BlinkDL/rwkv-4-raven",
        "title": "RWKV-4-Raven-14B-v8-Eng-20230408-ctx4096",
    },
    "raven-7b-ctx4096": {
        "repo_id": "BlinkDL/rwkv-4-raven",
        "title": "RWKV-4-Raven-7B-v7-Eng-20230404-ctx4096",
    },
    "raven-7b-ctx1024": {
        "repo_id": "BlinkDL/rwkv-4-pile-7b",
        "title": "RWKV-4-Pile-7B-Instruct-test4-20230326",
    },
    "rwkv-4-pile-169m": {
        "repo_id": "BlinkDL/rwkv-4-pile-169m",
        "title": "RWKV-4-Pile-169M-20220807-8023",
    },
    "raven-1b-ctx4096": {
        "repo_id": "BlinkDL/rwkv-4-raven",
        "title": "RWKV-4-Raven-1B5-v11-Eng99%-Other1%-20230425-ctx4096",
    },
    "raven-3b-ctx4096": {
        "repo_id": "BlinkDL/rwkv-4-raven",
        "title": "RWKV-4-Raven-3B-v11-Eng99%-Other1%-20230425-ctx4096",
    },
}


def fetch_tokenizer(tokenizer_path: Path, tokenizer_url):
    tokenizer_path.parent.mkdir(exist_ok=True)

    response = requests.get(tokenizer_url)
    tokenizer_path.write_bytes(response.content)

@hydra.main(version_base=None, config_path="conf", config_name="simple_rwkv")
def get_model_path(cfg : DictConfig):
    print(cfg)
    cache_dir = cfg.get('cache_dir') or Path('~/.cache/simple_rwkv').expanduser()
    tokenizer_url = cfg.model.get('tokenizer_url') or DEFAULT_TOKENIZER_URL
    tokenizer_path = cache_dir / "20B_tokenizer.json"
    if not tokenizer_path.exists():
        fetch_tokenizer(tokenizer_path, tokenizer_url)

    
    model_path = hf_hub_download(
        repo_id=cfg.model.repo_id, filename=f"{cfg.model.title}.pth",
        cache_dir=cache_dir
    )

    return model_path


if __name__ == "__main__":
    get_model_path()
