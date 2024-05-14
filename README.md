# MultiSuSiE
MultiSuSiE is a multi-ancestry extension of the [Sum of Single Effects model](https://github.com/stephenslab/susieR) (Wang et al. 2020 J. R. Statist. Soc. B, Zou et al. 2022 PLoS Genet.) implemented in Python. 

This MultiSuSiE implementation follows the [susieR](https://github.com/stephenslab/susieR) implementation as closely as possible. We thank the susieR developers for their work. 

For details of the method see our [preprint](https://www.medrxiv.org/content/10.1101/2024.05.13.24307291v1).

## Installation

There are 3 ways to install MultiSuSiE. The first two listed below will install a Python package and allow you to call MultiSuSiE via Python or a command line interface (in progress). The third (in progress) will only provide access to the command line interface.

### Installing into a fresh conda environment

The easiest way to install MultiSuSiE is to create a fresh conda environment.

To create a fresh conda environment capable of running multisusie run the following commands:
```
git clone https://github.com/jordanero/MultiSuSiE
cd MultiSuSiE
conda env create -f environment.yml
conda activate MultiSuSiE
pip install . -U
```

### PIP

It may be more convenient to use MultiSuSiE from your an existing conda environment. To do this, you'll need the following dependencies:
- Python >= 3.6
- numpy >= 1.20
- scipy >= 1.4.1
- tqdm 
- numba >= 0.51.0

It's posible that you can also get away with using older versions of these packages in some cases.

To install MultiSuSiE, activate the environment you'd like to install MultiSuSiE into (or don't if you'd like to install MultiSuSiE into your base environment) and run
```
git clone https://github.com/jordanero/MultiSuSiE
cd MultiSuSiE
pip install . -U
```

### Executable

(in progress)

## Running MultiSuSiE

The primary top-level MultiSuSie function is `multisusie_rss`. `multisusie_rss` accepts lists of numpy arrays containing GWAS summary statistics and LD matrices and performs the full fine-mapping algorithm. To see its full documentation, just start a Python session and type `import MultiSuSiE`, then `help(MultiSuSiES.multisusie_rss)`. 

The fastest way to get started is probably to check out examples/example.ipynb or examples/example.pdf, but we'll discuss the most important arguments for `multisusie_rss` at a high level below. Note that we'll denote the number of ancestries as K and number of variants as P.  

- b_list is a length K list (one per population) of length P numpy arrays representing GWAS effect sizes for each variant considered.
- s_list is a length K list (one per population) of length P numpy arrays representing GWAS effect size standard errors for each variant considered.
- R_list is a length K list (one per population) of PxP numpy arrays representing the LD correlation matrix for each population.
- varY_list is a length K representing the sample variance of the outcome in each population.
- population_sizes is a length K list of integers representing the GWAS sample size of each population.

## Questions?

Feel free to email jordanerossen@gmail.com with any questions.
