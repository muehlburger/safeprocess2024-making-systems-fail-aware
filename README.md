# Making Systems Fail-Aware

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23890438.svg)](https://doi.org/10.5281/zenodo.23890438)

Code for the paper "Making Systems Fail-Aware: A Semi-Supervised Machine Learning Approach for Identifying Failures by Learning the Correct Behavior of a System" by Herbert Muehlburger and Franz Wotawa (12th IFAC Symposium SAFEPROCESS 2024, IFAC-PapersOnLine 58(4), 7-12, [doi:10.1016/j.ifacol.2024.07.185](https://doi.org/10.1016/j.ifacol.2024.07.185)).

A Bi-LSTM forecaster learns the nominal behavior of a battery controller signal from healthy simulation data. Forecast errors above a threshold raise an alarm.

## Setup

The published models were written with Keras 2.15 under Python 3.11.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The plots use `text.usetex`, so a LaTeX installation is required.

## Usage

The project is a Kedro project; its package is called `mondev`. Pass the signal as a parameter (`temperature_degC_5`, `temperature_degC_6`, `volflow_l_min`, or `amperage_A`):

```bash
kedro run --pipeline=de --params="model_options.feature=temperature_degC_5"   # scale data
kedro run --pipeline=ds --params="model_options.feature=temperature_degC_5"   # train
kedro run --pipeline=me --params="model_options.feature=temperature_degC_5"   # evaluate
```

`./generate-results.sh` evaluates all four signals with the models in `data/06_models` and rebuilds `data/08_reporting/results_table.tex`. The `generate-results-*.sh` scripts do the same for the window-size tables. Neither retrains.

## Contents

| Path | Content |
| :-- | :-- |
| `conf/base/parameters.yml` | Epochs, learning rate, batch size, window size, split, thresholds |
| `src/mondev/pipelines/` | Pipelines `de` (data), `ds` (training), `me` (evaluation) |
| `src/mondev/utils.py` | Windowing, forecasting, metrics, plots, reports |
| `data/01_raw/` | Healthy and faulty simulation runs |
| `data/06_models/` | The four trained models and their training logs |
| `data/08_reporting/` | Published result tables |

## Reproducibility

Running `de` and `me` with the included models reproduces `data/08_reporting/results_table.tex`, including all alarm counts, up to floating-point noise in the CSV files. Retraining has no fixed seed and gives different models.

## Configuration

The thresholds are set per signal in `conf/base/parameters.yml`. The model is defined in `src/mondev/pipelines/data_science/nodes.py`.

## Citation

```bibtex
@article{muehlburger2024making,
  title   = {Making Systems Fail-Aware: A Semi-Supervised Machine Learning Approach for Identifying Failures by Learning the Correct Behavior of a System},
  author  = {M{\"u}hlburger, Herbert and Wotawa, Franz},
  journal = {IFAC-PapersOnLine},
  volume  = {58},
  number  = {4},
  pages   = {7--12},
  year    = {2024},
  doi     = {10.1016/j.ifacol.2024.07.185}
}
```

## Acknowledgements

This research is funded by the Austrian Federal Ministry of Climate Action, Environment, Energy, Mobility, Innovation and Technology (BMK) under the program "ICT of the Future", project 877587.

## License

[MIT](LICENSE)
