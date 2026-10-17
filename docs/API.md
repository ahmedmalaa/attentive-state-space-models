# Training and inference API

## Construct data

```python
import torch
from assm import SequenceBatch, AttentiveSSM, ModelConfig, TrainingConfig

# N patients, T padded visits, Dc continuous and Db binary features.
batch = SequenceBatch(
    continuous=torch.randn(12, 8, 2),
    binary=torch.randint(0, 2, (12, 8, 1)).float(),
    lengths=torch.tensor([8, 7, 6, 8, 5, 8, 7, 8, 6, 8, 7, 8]),
)
```

Continuous and binary tensors must be floating point with the same dtype and
device. Binary features must contain 0 or 1 at valid visits. Lengths must be
`torch.long` on the same device, between 1 and T. Only right-padding is supported.
Padded entries, including NaNs there, are replaced by zero; valid NaNs/infinities
raise an error. Observations are not automatically standardized.

`batch.select(indices)` selects patients; pass a list/tensor to keep the batch
dimension. `batch.prefix(stop)` retains visits 0 through stop-1 and caps lengths.
`batch.to(device)` moves all tensors. A model and its batches must have matching
devices/dtypes; the default model uses float32. Use `.model.double()` if your
batch intentionally uses float64.

For continuous-only data set `n_binary=0` and pass `torch.empty(N,T,0)` with the
same dtype and device. Binary-only data is not supported by this implementation.

## Fit

```python
model = AttentiveSSM(
    ModelConfig(num_states=3, n_continuous=2, n_binary=1,
                hidden_dim=32, num_layers=1, covariance="full"),
    device="cpu", seed=19,
)
model.fit(train_batch, validation=validation_batch,
          training=TrainingConfig(epochs=60, batch_size=64, num_particles=2))
```

`fit` returns the estimator. Inputs contain observations/lengths only. Optional
validation contributes logged losses but no updates. The warm start is enabled
by default and uses only training observations. To disable it, set
`TrainingConfig(warm_start=False)`.

Training uses the paper's Monte Carlo variational objective with a categorical
score estimator. An epoch consists of `ceil(N / batch_size)` patient minibatches
sampled uniformly with replacement. Logged negative ELBO values are normalized
per valid visit. `history_` is a list of epoch dictionaries; `mixture_` contains
the initialization GMM when used; `initialization_history_` records its guide
pretraining losses. `training_seconds_` measures SVI only. The notebook also
measures total initialization-plus-SVI time.

Calling fit again creates a new optimizer. To retain current weights, set
`warm_start=False`; otherwise initialization runs again. In each fit, a scoped
Pyro parameter store prevents accidental sharing with another estimator.

## Smooth complete records

```python
result = model.posterior(batch, num_samples=128)
```

| Field | Shape | Meaning |
|---|---|---|
| `probs` | `[N,T,K]` | Monte Carlo guide marginals, zero in padding |
| `states` | `[N,T]` | Marginal argmax, -1 in padding |
| `samples` | `[S,N,T]` | Whole guide-sampled histories, -1 in padding |
| `attention` | `[N,T,T]` | Transition weights; first row and padded target rows are zero |

These are approximate *smoothed* estimates: the backward LSTM sees all supplied
visits. They are not exact posterior probabilities or a Viterbi path. Marginal
probabilities average guide conditionals over sampled previous states, which
reduces sampling variance. Use `pyro.set_rng_seed(...)` before a call to reproduce
its Monte Carlo draws. No inference method updates parameters.

## Filter prefixes

```python
online = model.filter(batch, num_samples=128)
current = model.posterior(batch.prefix(8))["probs"][:, -1]
```

`filter` returns `probs [N,T,K]` and `states [N,T]` using only the prefix available
at each target visit. It reruns the guide at each prefix, so it costs more than
one smoothing pass. For unequal histories, gather each patient's last valid
visit using `lengths - 1`; `[:, -1]` above assumes that the prefix has eight valid
visits for every patient. This is amortized prefix inference, not exact filtering.

## Forecast

```python
future = model.forecast(history, horizon=3, num_samples=128)
```

Requires equal unpadded history lengths. Group by length or process patients
individually otherwise. Historical states are sampled from the guide; future
states and observations are then sampled from the fitted generative model.
Generated observations are fed back to attention at the next step.

| Field | Shape |
|---|---|
| `states` | `[S,N,H]` |
| `continuous` | `[S,N,H,Dc]` |
| `binary` | `[S,N,H,Db]` |
| `state_probs` | `[N,H,K]` |

Quantiles of continuous draws are observation predictive intervals under the
point-estimate model; parameter uncertainty is not represented.

## Inspect and save

```python
P = model.model.transition_matrix.detach()
means = model.model.emissions.means.detach()
covariances = model.model.emissions.covariance.detach()
alpha = model.attention_weights(batch)
model.save("fitted_assm.pt")
reloaded = AttentiveSSM.load("fitted_assm.pt", device="cpu")
```

Checkpoints contain model config, tensors and training loss history. They do not
contain the initialization GMM, optimizer state or baseline buffers, and are
intended for inference. Only load checkpoints from sources you trust; the loader
uses PyTorch's `weights_only=True` mode.

## Tiny exact diagnostic

```python
exact = model.exact_posterior(batch.select([0]).prefix(5))
```

Returns `probs [1,T,K]` and scalar `log_evidence`, enumerating every latent path.
Requires one unpadded trajectory and refuses more than `max_paths=100_000`.
This exponential calculation verifies approximation error for a small example;
it is not the training algorithm or the normal inference path.
