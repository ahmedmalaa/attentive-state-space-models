# Attentive state-space modeling with Pyro

A commented, modular implementation of **Attentive State-Space Modeling of
Disease Progression**, Alaa & van der Schaar, NeurIPS 2019. The package implements
the paper's non-Markovian state transitions, LSTM encoder/decoder attention,
mixed Gaussian/conditional-Bernoulli emissions, and structured variational guide
with shared attention. The notebook generates its own data and measures hidden
state recovery on held-out patients.

## Run in Google Colab

1. Download `attentive_ssm_colab.zip` and `demo.ipynb`.
2. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload
   notebook**, and open `demo.ipynb`. You can also extract the ZIP and upload the
   copy of the notebook inside it.
3. Choose a **CPU** runtime and **Runtime → Run all**.
4. When the first cell asks, upload `attentive_ssm_colab.zip`. It extracts the
   modules and installs dependencies, reusing a compatible existing PyTorch.

No Google Drive mount, repository clone, credentials, or external dataset is
required. The delivered notebook includes actual executed outputs. Running all
cells trains a fresh model; it does not merely load the delivered checkpoint.
The default experiment uses 480 patients, 10–16 visits, 3 states, and 60 SVI
epochs. Training takes several minutes on a CPU; hardware affects runtime.

## Run locally

From the extracted `attentive_ssm_colab` directory:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_demo.py
```

`run_demo.py` executes and updates `demo.ipynb`. In restricted environments that
do not allow Jupyter kernel sockets, use `python run_demo.py --in-process`. That
mode executes the same cells through IPython and records real rich outputs and
figures. You can also open the notebook in your usual Jupyter frontend.

To import the package from outside this directory, install it with
`python -m pip install -e .`. Python 3.10+ is required. The dependency ranges
allow recent Colab installations; `results/metrics.json` records the versions
used for the delivered verification run. Pyro is pinned to 1.9.1.

## Folder contents

| File | Responsibility |
|---|---|
| `demo.ipynb` | End-to-end walkthrough, synthetic cohort, training, evaluation and API examples |
| `assm/config.py` | Model and training configuration dataclasses |
| `assm/data.py` | Validated patient batches, padding, prefixes and device transfer |
| `assm/networks.py` | Seq2seq attention, backward LSTM and linear combiner |
| `assm/emissions.py` | Full/diagonal Gaussian and conditional Bernoulli emissions |
| `assm/model.py` | Pyro generative model and structured categorical guide |
| `assm/training.py` | Observation-only warm start and score-function SVI |
| `assm/inference.py` | Smoothing, prefix filtering, forecasting and tiny exact diagnostic |
| `assm/estimator.py` | Public `fit`, `posterior`, `filter`, `forecast`, `save`, `load` API |
| `assm/synthetic.py` | Known non-Markovian synthetic data generator |
| `assm/evaluation.py` | Permutation alignment and recovery metrics |
| `docs/METHOD.md` | Equation map, design choices and limitations |
| `docs/API.md` | Training/inference interface and tensor shape contracts |
| `tests/test_probabilistic_contract.py` | Focused probabilistic and leakage checks |
| `results/` | Executed metrics, training history and inference checkpoint |
| `figures/` | Exported notebook plots |

## Minimal API

```python
from assm import AttentiveSSM, ModelConfig, TrainingConfig, generate_synthetic

data = generate_synthetic()["batch"]
train, test = data.select(list(range(320))), data.select(list(range(384, 480)))
model = AttentiveSSM(ModelConfig(num_states=3), seed=19)
model.fit(train, training=TrainingConfig(epochs=60))  # No true states passed.
posterior = model.posterior(test, num_samples=128)
state_probabilities = posterior["probs"]  # [patients, visits, states]

# A common-length prefix supplies only observations available at that time.
prediction = model.forecast(test.prefix(8), horizon=3, num_samples=128)
model.save("my_model.pt")
restored = AttentiveSSM.load("my_model.pt")
```

## What is demonstrated

The notebook reports state accuracy, adjusted Rand index, state probability
log loss, emission-mean recovery, a confusion matrix, trajectories, and a short
exact-posterior comparison. It separately reports the observation-only GMM and
the initialized guide, so emission separation and warm starting are visible.
True states are used only for evaluation/illustration. Train, validation and test
patients are disjoint; validation is monitored but never fitted. State labels
are aligned by emission means for scoring, not by test labels or gradient updates.

This is an independent implementation, not the authors' original code or a
reproduction of the clinical experiments. Material choices are documented:
strictly past-only attention; a smaller demo network; optional observation-only
initialization; Adam updates for emission parameters; and a discrete
score-function estimator. Strong synthetic state recovery does not establish
unique attention recovery, clinical validity, or superiority over a trained HMM.

## References

- Alaa, A. M., & van der Schaar, M. (2019).
  [Attentive State-Space Modeling of Disease Progression](https://proceedings.neurips.cc/paper_files/paper/2019/hash/1d0932d7f57ce74d9d9931a2c6db8a06-Abstract.html).
- Pyro: [SVI and TraceGraph_ELBO](https://docs.pyro.ai/en/stable/inference_algos.html).
- Pyro: [ELBO gradient estimators](https://pyro.ai/examples/svi_part_iii.html).
