import tomllib
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .schemas import Experiment, RawData


def get_experimental_data(config: Experiment) -> dict[str, torch.Tensor]:

    local_dir = Path("./data").resolve()
    file = local_dir / f"{config.model}_{config.data}.pt"

    if file.exists():
        return torch.load(file)

    print("\nCouldn't find the file, downloanding from Hugging face. \n")

    url_file = local_dir / "data_links.toml"
    if not url_file.exists():
        raise FileNotFoundError(
            "Couldn't locate the TOML file which contains urls for downloading dataset"
        )

    with url_file.open("rb") as urf:
        url = tomllib.load(urf)
        info = url[config.data][config.model]

    try:
        repo_id = info["repo_id"]
        filename = info["filename"]
    except KeyError:
        raise KeyError(
            f"Entry '{config.model} -> {config.data}' doesn't exist in the TOML config."
        )

    downloaded_str = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir, repo_type="dataset"
    )
    return torch.load(downloaded_str)


def user_loader(adr: Path) -> RawData:
    raise NotImplementedError()
