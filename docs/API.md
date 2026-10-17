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

Note that the warm start by fitting a Gaussian mixture model is enabled by default and uses only training observations. To disable it, set `TrainingConfig(warm_start=False)`. Training uses the Monte Carlo variational objective in our paper with a categorical score estimator. Every epoch consists of `ceil(N / batch_size)` patient minibatches sampled uniformly with replacement. Also note that the logged negative ELBO values are normalized per visit (`history_` is the list of epoch dictionaries).  `mixture_` contains the initialization GMM when used, and `initialization_history_` records its pretraining losses before applying the SVI training process. 

## Posterior inference

```python
result = model.posterior(batch, num_samples=128)
```

| Field | Shape | Description |
|---|---|---|
| `probs` | `[N,T,K]` | Monte Carlo guide marginals, zero in padding |
| `states` | `[N,T]` | Marginal argmax, -1 in padding |
| `samples` | `[S,N,T]` | Whole guide-sampled histories, -1 in padding |
| `attention` | `[N,T,T]` | Transition weights; first row and padded target rows are zero |


## Filtering

```python
online = model.filter(batch, num_samples=128)
current = model.posterior(batch.prefix(8))["probs"][:, -1]
```

`filter` returns `probs [N,T,K]` and `states [N,T]` using only the data available
at the current visit. 

## Forecasting

```python
future = model.forecast(history, horizon=3, num_samples=128)
```

Here, states are sampled from the inference network and future states and observations are then sampled from the fitted generative model.
Generated observations are fed back to attention at the next step.

| Field | Shape |
|---|---|
| `states` | `[S,N,H]` |
| `continuous` | `[S,N,H,Dc]` |
| `binary` | `[S,N,H,Db]` |
| `state_probs` | `[N,H,K]` |


## Saving and loading model checkpoints

```python
P = model.model.transition_matrix.detach()
means = model.model.emissions.means.detach()
covariances = model.model.emissions.covariance.detach()
alpha = model.attention_weights(batch)
model.save("fitted_assm.pt")
reloaded = AttentiveSSM.load("fitted_assm.pt", device="cpu")
```

