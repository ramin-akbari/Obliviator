# Obliviator

Obliviator is a highly optimized, PyTorch-based library for iterative concept erasure in foundation models (e.g., DeepSeek, LLaMA, GPT-2, BERT). It utilizes HSIC (Hilbert-Schmidt Independence Criterion) penalties followed by solving a constrained optimization in RKHS to smoothly remove unwanted semantic attributes while preserving model utility. See [`Obliviator`](https://openreview.net/pdf?id=GcjpjIHDZn) on open review (NeurIPS 2025).

This repository serves two purposes:
1. A standalone **Python library** for integrating concept erasure into your own pipelines.
2. An **experimental framework** to replicate the paper's specific methodology and ablation studies.

## 📦 Installation

Obliviator is built and managed using [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver. 

### Option 1: Using Obliviator as a Library
If you simply want to import `obliviator.Supervised` or `obliviator.Unsupervised` in your own independent project, you can install it directly from GitHub:

```bash
uv add git+https://github.com/ramin-akbari/obliviator.git
```

### Option 2: Cloning for Development & Reproducing Experiments
If you want to run `main.py`, modify the data loaders, or reproduce the experimental setup, you must clone the repository and sync the complete environment.


- Clone the repository
```bash
git clone git@github.com:ramin-akbari/Obliviator.git
```
- Create a virtual environment and sync dependencies
```bash
cd obliviator
uv sync
```
- you can either activate the virtual env by `source .venv/bin/activate` or use `uv run <file_name>`

### Using the Built-in CLI Manual
Obliviator is powered by [`tyro`](https://brentyi.github.io/tyro/), which automatically generates comprehensive documentation for all hyperparameter configurations directly in your terminal.

To see the available top-level commands (reproducing experiments vs. using custom data), run:

```bash
uv run main.py --help
```

```bash
uv run main.py expr --help
```

```bash
uv run main.py sup --help
```

### 🚀 Reproducing Experiments
The experimental pipeline handles data downloading (via Hugging Face) and iterative erasure. 
***Basic Syntax***

```bash
uv run main.py expr --model <MODEL> --data <DATASET> --mode <sup|unsup>
```

***Examples***

- Unsupervised concept erasure on GPT-2 using the Bios dataset
```bash
uv run main.py expr --model gpt2 --data bios --mode unsup
```
- Supervised concept erasure on LLaMA using the Dial-Sen dataset

```bash
uv run main.py expr --model llama --data dial-sen --mode sup 
```

Note: The CLI automatically pulls the corresponding hyperparameter configurations defined in configs/defaults.py.

### 💻 Using Custom Data
You can also pass your own datasets through the CLI using the sup or unsup subcommands. see `configs/user.py` and `configs/loader.py` for more details.
