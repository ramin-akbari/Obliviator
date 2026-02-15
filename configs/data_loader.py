import tomllib
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .schemas import ErasureData, Experiment


def get_experimental_data(config: Experiment) -> dict[str, torch.Tensor]:

    local_dir = Path("../data").resolve()
    file = local_dir / f"{config.data}_{config.mode}.pt"
    if file.exists():
        return torch.load(file)

    print("Couldn't find the file, downloanding from Hugging face. \n")

    url_file = local_dir / "data_links.toml"
    if not url_file.exists():
        raise FileNotFoundError("Couldn't locate the url for downloading the file")

    with url_file.open("rb") as urf:
        url = tomllib.load(urf)
        info = url[config.model][config.data]

    try:
        repo_id = info["repo_id"]
        filename = info["filename"]
    except KeyError:
        raise KeyError(f"Entry '{config.model} -> {config.data}' doesn't exis in TOML.")

    downloaded_str = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir
    )
    return torch.load(downloaded_str)


def user_loader(adr: Path) -> ErasureData:
    raise NotImplementedError()
