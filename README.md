# MultiSuSiE
MultiSuSiE is a multi-ancestry extension of the [Sum of Single Effects model](https://github.com/stephenslab/susieR) (Wang et al. 2020 J. R. Statist. Soc. B, Zou et al. 2022 PLoS Genet.) implemented in Python. 

This MultiSuSiE implementation follows the [susieR](https://github.com/stephenslab/susieR) implementation as closely as possible. We thank the susieR developers for their work. 

Please cite:

Rossen, J., Shi, H., Strober, B.J. et al. MultiSuSiE improves multi-ancestry fine-mapping in All of Us whole-genome sequencing data. Nat Genet 58, 67–76 (2026). https://doi.org/10.1038/s41588-025-02450-5

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

## Running MultiSuSiE on binary traits

We recommend only applying MultiSuSiE to binary traits when the MAF, case fraction, and sample size conditions provided in Loh et al. 2018 Nature Genetics are met (see p. 10-11, 24 of the [supplement](https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0144-6/MediaObjects/41588_2018_144_MOESM1_ESM.pdf)).

When running MultiSuSiE on binary traits, we recommend providing `b_list`, `s_list`, and `varY_list`, not `z_list` and setting `var_y = residual_variance = 1/(phi*(1-phi))` with `phi = n_cases/total_n` (separately for each ancestry). This suggestion was initially provided by the SuSiE authors [here](https://github.com/stephenslab/susieR/issues/74). Many thanks to Sophia Gunn for bringing this issue to our attention.


## Questions?

Feel free to open an issue on GitHub (preferred) or email jordanerossen@gmail.com.

## Numerical precision and convergence

RSS defaults to `float_type=np.float64`; explicit float32 remains available
and uses about half the array memory. Posterior moments use direct precision
solves to avoid cancellation. Both APIs require a finite `0 <= ELBO change < tol`,
matching [updated SuSiE](https://github.com/stephenslab/susieR/blob/9c406e785ad002a5121cb79acd1fedd4ece10173/R/model_methods.R).
ELBO decreases do not trigger convergence; fitting can continue to `max_iter`.
Numerical stability does not correct LD mismatch or establish global optimality.

## Development checks

To run the same quality checks locally that run in CI:

```bash
pip install -e ".[test]"
ruff check .
pytest -q
```
