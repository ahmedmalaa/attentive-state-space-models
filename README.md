# Attentive State-space Modeling of Disease Progression

This is a [pyro](https://pyro.ai/) implementation of [*Attentive State-Space Modeling of Disease Progression*](https://proceedings.neurips.cc/paper_files/paper/2019/file/1d0932d7f57ce74d9d9931a2c6db8a06-Paper.pdf) by Alaa and van der Schaar in NeurIPS 2019. This codebase implements the non-Markovian/attention-based state transitions, LSTM encoder/decoder attention, mixed Gaussian/conditional-Bernoulli emissions, and structured variational inference using built-in functions in [pyro](https://pyro.ai/). The notebook "Attentive SS Models.ipynb" contains a demo of how to use the model training and inference APIs on a toy synthetic data generation process.

## Installation

You can install all requirements for this codebase through:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

To import the package from outside this directory, you can install it with
`python -m pip install -e .`. Note that Python 3.10+ is required. 

## Folder contents

| File | Responsibility |
|---|---|
| `Attentive SS Models.ipynb` | Demo in synthetic data with training, evaluation and API examples |
| `assm/config.py` | Model and training configuration dataclasses |
| `assm/networks.py` | Seq2seq attention, backward LSTM and linear combiner implementations |
| `assm/emissions.py` | Full/diagonal Gaussian and conditional Bernoulli emissions |
| `assm/model.py` | Pyro generative model and structured inference network |
| `assm/training.py` | Warm starting and implementation of the score function SVI |
| `assm/inference.py` | Smoothing, filtering, and forecasting functions |
| `assm/estimator.py` | Different functions for the model class, e.g. `fit`, `posterior`, `filter`, `forecast`, `save`, `load` API |
| `assm/synthetic.py` | Data generation model for the synthetic data experiment in the demo |
| `assm/data.py` | Utils for dealing with patient batches, padding, prefixes and device transfer |
| `assm/evaluation.py` | Eval metrics |
| `docs/METHOD.md` | Model description |
| `docs/API.md` | Training/inference interface and required tensor shapes|
| `tests/test_probabilistic_contract.py` | Checks for known errors |
| `results/` | Our eval metrics, training logs and inference checkpoint |
| `figures/` | Exported plots from tge demo |

## Minimal API

You can find more detailed description of the API in this [link](https://github.com/ahmedmalaa/attentive-state-space-models/edit/master/docs/API.md).

```python
from assm import AttentiveSSM, ModelConfig, TrainingConfig, generate_synthetic

# load synthetic data
data = generate_synthetic()["batch"]
train, test = data.select(list(range(320))), data.select(list(range(384, 480)))

# instantiate and train an attentive state space model
model = AttentiveSSM(ModelConfig(num_states=3), seed=19)
model.fit(train, training=TrainingConfig(epochs=60))

# latent state inference
posterior = model.posterior(test, num_samples=128)
state_probabilities = posterior["probs"]  # [patients, visits, states]

# prediction and forecasting of future states and observations
prediction = model.forecast(test.prefix(8), horizon=3, num_samples=128)
model.save("my_model.pt")
restored = AttentiveSSM.load("my_model.pt")
```

## References

- Alaa, A. M., & van der Schaar, M. (2019).
  [Attentive State-Space Modeling of Disease Progression](https://proceedings.neurips.cc/paper_files/paper/2019/hash/1d0932d7f57ce74d9d9931a2c6db8a06-Abstract.html).
- Pyro: [SVI and TraceGraph_ELBO](https://docs.pyro.ai/en/stable/inference_algos.html).
- Pyro: [ELBO gradient estimators](https://pyro.ai/examples/svi_part_iii.html).
