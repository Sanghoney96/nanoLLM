# nanoLLM

## Overview

nanoLLM is a lightweight LLM training framework **implemented from scratch** using PyTorch.  
The goal of this project is to provide a minimal yet practical environment for training and experimenting with Transformer-based language models, inspired by projects such as nanoGPT.

The framework is designed to expose the core components of the language model training pipeline rather than hiding them behind heavy abstractions. It includes modular building blocks such as a tokenizer, dataset module, model architecture, training loop, and text generation utilities.

Each component is implemented with clear and independent abstractions so that new models, tokenizers, or training strategies can be easily integrated.  
By focusing on simplicity and modularity, nanoLLM aims to provide a **clean and extensible architecture** for learning, experimenting with, and prototyping Transformer-based language models.

## Key Features

- **Modular architecture for building LLM training pipelines**
- **Extensible component design for easily integrating custom models and tokenizers**
- **Minimal external dependencies to keep the framework lightweight**
- **A clear and structured training pipeline suitable for educational and experimental use**