# MACE Graph Neural Network

This repository contains a course-focused implementation of the **MACE** (Multiplicative Atomic Cluster Expansion) graph neural network for atomistic machine learning.

The implementation will use [cuEquivariance](https://github.com/NVIDIA/cuEquivariance) to accelerate equivariant operations on NVIDIA GPUs while preserving the rotational and translational symmetries required by molecular and materials simulations.

## Goals

- Implement the core MACE architecture and message-passing workflow.
- Learn how equivariant graph neural networks represent atomic environments.
- Benchmark cuEquivariance-accelerated operations.
- Train and evaluate models on atomistic datasets.

## Status

This project is under development as part of a special course. Documentation, installation instructions, examples, and experiments will be added as the implementation progresses.

## Project structure

```text
configs/      Training configurations
data/         Data
modules/      Basic modules for the model
scripts/      Training and evaluation scripts
tests/        Unit and integration tests
```

## Requirements

- Python
- PyTorch
- CUDA-enabled NVIDIA GPU
- cuEquivariance

Detailed setup instructions will be provided once the project structure is established.
