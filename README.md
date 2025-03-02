# Neural Hydrology

A Python library for training neural networks with a focus on hydrological applications. This repository is a fork of the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) project, customized for specific research needs.

## Overview

NeuralHydrology is built on top of [PyTorch](https://pytorch.org/) and emphasizes modularity to facilitate:
- Easy integration of new datasets
- Implementation of new model architectures
- Customization of training aspects (loss functions, optimizers, regularization)
- Configuration-based training without code modification

## Key Features

- **Modular Design**: Easily extend and customize components
- **Configuration-Driven**: Train models using configuration files without code changes
- **Research-Focused**: Built for flexibility and experimentation in hydrological applications
- **PyTorch Foundation**: Leverages PyTorch's powerful deep learning capabilities

## Getting Started

1. Clone this repository
2. Install dependencies (requirements will be listed)
3. Configure your experiment using the `config.yml` file
4. Run training using the provided scripts

## Configuration

The `config.yml` file controls all aspects of model training, including:
- Data configuration (datasets, inputs, targets)
- Model architecture and parameters
- Training settings (batch size, learning rate, etc.)
- Validation and testing parameters

## Original Project

This repository is based on the original NeuralHydrology project by the AI for Earth Science group at the Institute for Machine Learning, Johannes Kepler University, Linz, Austria.

For the original project:
- Documentation: [neuralhydrology.readthedocs.io](https://neuralhydrology.readthedocs.io)
- Research Blog: [neuralhydrology.github.io](https://neuralhydrology.github.io)
- Issues & Features: [GitHub Issues](https://github.com/neuralhydrology/neuralhydrology/issues)

## Citation

If you use this work in your research, please cite the original JOSS paper:

```bibtex
@article{kratzert2022joss,
  title = {NeuralHydrology --- A Python library for Deep Learning research in hydrology},
  author = {Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz},
  journal = {Journal of Open Source Software},
  publisher = {The Open Journal},
  year = {2022},
  volume = {7},
  number = {71},
  pages = {4050},
  doi = {10.21105/joss.04050},
  url = {https://doi.org/10.21105/joss.04050},
}
```

For questions about the original NeuralHydrology project:
- Use the [discussion section](https://github.com/neuralhydrology/neuralhydrology/discussions)
- Open an [issue](https://github.com/neuralhydrology/neuralhydrology/issues)
- Email: neuralhydrology(at)googlegroups.com
