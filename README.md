# 🤗 nanoLLM

> Minimal LLM framework built from scratch in Pytorch

nanoLLM is a lightweight LLM training framework **implemented from scratch** using PyTorch.  
The goal of this project is to provide a minimal yet practical environment for training and experimenting with Transformer-based language models, inspired by projects such as nanoGPT.

The framework is designed to expose the core components of the language model training pipeline rather than hiding them behind heavy abstractions. It includes modular building blocks such as a tokenizer, dataset module, model architecture, training loop, and text generation utilities.

Each component is implemented with clear and independent abstractions so that new models, tokenizers, or training strategies can be easily integrated.  
By focusing on simplicity and modularity, nanoLLM aims to provide a **clean and extensible architecture** for learning, experimenting with, and prototyping Transformer-based language models.

## 🔑 Key Features

- **Modular architecture for building LLM training pipelines**
- **Extensible component design for easily integrating custom models and tokenizers**
- **Minimal external dependencies to keep the framework lightweight**
- **A clear and structured training pipeline suitable for educational and experimental use**

## Installation

This project uses **uv** for dependency management.

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/nanollm.git
cd nanollm
```

### 2. Install dependencies

Install all dependencies and create a virtual environment:

```bash
uv sync
```

This command will:

* create a virtual environment (`.venv`)
* install all required dependencies from `pyproject.toml`
* reproduce the exact environment defined by `uv.lock`

### 3. Activate the environment

If you want to manually activate the environment:

```bash
source .venv/bin/activate
```

---

## Environment Reproducibility

The environment is fully defined by:

* `pyproject.toml`
* `uv.lock`

Any user can reproduce the exact same environment with:

```bash
uv sync
```
