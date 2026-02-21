import tomllib
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .schemas import Expr, RawData


def user_loader(adr: Path) -> RawData:
    raise NotImplementedError()


def load_experimental_data(config: Expr) -> RawData:
    local_dir = Path("./data").resolve()
    file = local_dir / f"{config.model}_{config.data}.pt"

    if file.exists():
        return RawData(**torch.load(file))

    print("\nCouldn't locate the file, downloanding from Hugging face ... \n")

    url_file = local_dir / "data_links.toml"
    if not url_file.exists():
        raise FileNotFoundError(
            "Couldn't locate the TOML file which contains urls for downloading dataset. \n"
        )

    with url_file.open("rb") as urf:
        url = tomllib.load(urf)

    try:
        info = url[config.data][config.model]
    except KeyError:
        raise KeyError(
            f"Entry '{config.model} -> {config.data}' doesn't exist in the TOML config."
        )
    downloaded_str = hf_hub_download(
        repo_id=info["repo_id"],
        filename=info["filename"],
        local_dir=local_dir,
        repo_type="dataset",
    )
    return RawData(**torch.load(downloaded_str))
