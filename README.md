# MultiSuSiE
MultiSuSiE is a multi-ancestry extension of the [Sum of Single Effects model](https://github.com/stephenslab/susieR) (Wang et al. 2020 J. R. Statist. Soc. B, Zou et al. 2022 PLoS Genet.) implemented in Python. 

This MultiSuSiE implementation follows the [susieR](https://github.com/stephenslab/susieR) implementation as closely as possible. We thank the susieR developers for their work. 

## Getting Started

MultiSuSiE has the following dependencies:
- Python >= 3.6
- numpy >= 1.20
- scipy >= 1.4.1
- tqdm 
- numba >= 0.51.0
It's posible that you can also get away with using older versions of these packages in some cases.

There are 3 ways to install MultiSuSiE

### Installing into a fresh conda environment

The easiest way to install MultiSuSiE is to create a fresh conda environment. However, it may be more convenient to be able to run MultiSuSiE from your favorite existing conda environment.

To create a fresh conda environment capable of running multisusie:
```
git clone https://github.com/jordanero/MultiSuSiE
cd MultiSuSiE
conda env create -f MultiSuSiE.yml
conda activate MultiSuSiE
pip install .
```

### PIP

To install MultiSuSiE into an existing environment, activate the environment (or dont if you'd like to install into your base environment) and run
```
git clone https://github.com/jordanero/MultiSuSiE
cd MultiSuSiE
pip install .
```

### Executable

## Tutorial

The fastest way to get started is to check out 

