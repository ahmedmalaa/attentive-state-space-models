# Verified example run

The 13 notebook code cells were executed in order through the provided IPython
in-process runner, including the full 60-epoch training run. The environment
does not permit Jupyter kernel sockets, so execution used `python run_demo.py
--in-process`. The code has not been executed inside Google's hosted Colab
service. The notebook's Colab setup installs dependencies and uploads the
accompanying ZIP without requiring a private data source.

## Recorded results

| Quantity | Result |
|---|---:|
| Held-out patients | 96 |
| Held-out valid visits | 1267 |
| Smoothed hidden-state accuracy | 98.3425% |
| Adjusted Rand index | 0.9487 |
| Filtering accuracy | 98.1847% |
| Observation-only GMM accuracy | 98.0268% |
| Initialized-guide accuracy | 98.2636% |
| Emission-mean RMSE | 0.0402 |
| Transition-matrix MAE | 0.0608 |
| Guide/exact marginal MAE on 5 visits | 0.0028 |
| Initialization plus SVI wall time | 226.4 seconds |

The strong GMM baseline shows that most latent-state separation is already
available in the biomarkers. The SVI-trained temporal model improves the state
probability log loss and slightly improves accuracy in this run. This is not
a multi-seed comparison against a fitted HMM or a clinical validation study.

## Verification performed

- All 10 focused probabilistic tests passed (`python -m pytest -q`).
- Every notebook code cell ran without an exception.
- Six figures were generated and visually inspected.
- The Pyro joint log density agrees with the explicit mathematical factorization.
- Model and guide receive finite nonzero gradients through the discrete ELBO.
- Future observations do not affect earlier transition attention or filtering.
- The inference checkpoint reload reproduces posterior probabilities under a fixed seed.
- Continuous-only/diagonal and mixed/full-covariance paths are covered.

Runtime versions and numerical outputs are in `metrics.json`; the exact
per-epoch losses are in `training_history.csv`. The notebook preserves its
executed figures and outputs. Different versions, devices and seeds may change
the numerical result.
