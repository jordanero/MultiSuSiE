# MultiSuSiE
MultiSuSiE is a multi-ancestry extension of the [Sum of Single Effects model](https://github.com/stephenslab/susieR) (Wang et al. 2020 J. R. Statist. Soc. B, Zou et al. 2022 PLoS Genet.) implemented in Python. 

This MultiSuSiE implementation follows the [susieR](https://github.com/stephenslab/susieR) implementation as closely as possible. We thank the susieR developers for their work. 

For details of the method see our [preprint](https://www.medrxiv.org/content/10.1101/2024.05.13.24307291v1).

## Installation

There are 2 ways to install MultiSuSiE. MultiSuSiE is a Python package and does not have a command-line interface. We hope to add a command-line interface in the future.

### Installing into a fresh conda environment

The easiest way to install MultiSuSiE is to create a fresh conda environment.

To create a fresh conda environment capable of running MultiSuSiE run the following commands:
```
git clone https://github.com/jordanero/MultiSuSiE
cd MultiSuSiE
conda env create -f environment.yml
conda activate MultiSuSiE
pip install . -U
```


### Installing into an existing environment

It may be more convenient to use MultiSuSiE in an existing conda environment or python virtual environment. To do this, you'll need the following dependencies:
- Python >= 3.8
- numpy >= 1.20
- scipy >= 1.4.1
- tqdm 
- numba >= 0.51.0

It's posible that you can get away with using older versions of these packages in some cases.

To install MultiSuSiE, activate the environment you'd like to install MultiSuSiE into (or don't if you'd like to install MultiSuSiE into your base environment) and run
```
pip install git+https://github.com/jordanero/MultiSuSiE.git
```

We've also added an install script that attempts to install the necessary dependencies (other than Python >= 3.8), install MultiSuSiE, and run an example. The install script can be run with the following commands
```
git clone https://github.com/jordanero/MultiSuSiE
bash MultiSuSiE/install.sh
```

## Running MultiSuSiE

The primary top-level MultiSuSie function is `multisusie_rss`. `multisusie_rss` accepts lists of numpy arrays containing GWAS summary statistics and LD matrices and performs the full fine-mapping algorithm. To see its full documentation, just start a Python session and type `import MultiSuSiE`, then `help(MultiSuSiE.multisusie_rss)`. 

The fastest way to get started is probably to check out [the example](https://github.com/jordanero/MultiSuSiE/blob/main/examples/example.ipynb) (seems like there are currently issues rendering .ipynb files on github, so if you get an error, view the pdf in the same directory), but we'll discuss the most important arguments for `multisusie_rss` at a high level below. Note that we'll denote the number of ancestries as K and number of variants as P.  

- b_list is a length K list (one per population) of length P numpy arrays representing GWAS effect sizes for each variant considered.
- s_list is a length K list (one per population) of length P numpy arrays representing GWAS effect size standard errors for each variant considered.
- R_list is a length K list (one per population) of PxP numpy arrays representing the LD correlation matrix for each population.
- varY_list is a length K representing the sample variance of the outcome in each population.
- population_sizes is a length K list of integers representing the GWAS sample size of each population.

A MultiSuSiE call to `multisusie_rss` will look something like this:
```
ss_fit = MultiSuSiE.multisusie_rss(
    b_list = beta_hat_list,
    s_list = se_list,
    R_list = R_list,
    varY_list = varY_list,
    population_sizes = N_list,
)
```

## Running MultiSuSiE faster

MultiSuSiE with summary statistics (`multisusie_rss`) runtime and memory requirements can be drastically improved by setting `low_memory_mode = True`. This parameter is not enabled by default because the input summary statistic and LD matrix numpy arrays will be mutated over the course of function evaluation and will not be returned to their initial state. If you understand this, we recommend setting `low_memory_mode = True`.

## Questions?

Feel free to open an issue on GitHub (preferred) or email jordanerossen@gmail.com.
