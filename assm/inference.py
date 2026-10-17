"""Pyro implementation of Attentive State-Space Models 
Code for sampling, marginal inference and forecasting with Attentive SS models.

"""

from itertools import product
import torch
from torch.nn import functional as F
import pyro.distributions as dist

@torch.no_grad()
def posterior(model, batch, num_samples=128):
    """Draw whole paths from q and average conditionals for marginal probabilities.

    Returns probs [N,T,K], states [N,T], samples [S,N,T], attention [N,T,T].
    Probs and attention are zero in padding; decoded/sample states are -1.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    model.validate_batch(batch)
    model.eval()

    n, tmax = batch.n_patients, batch.n_visits
    alpha   = model.attention(batch.context)
    h       = model.guide_network.backward(batch.context, batch.mask)

    expanded_alpha        = alpha.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(num_samples * n, tmax, tmax)
    expanded_h            = h.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(num_samples * n, tmax, -1)
    states, probabilities = [], []

    for t in range(tmax):
        forward = model.forward_probs(expanded_alpha[:, t], states)
        q       = model.guide_network.logits(expanded_h[:, t], forward).softmax(-1)
        
        states.append(dist.Categorical(probs=q).sample())
        # E_{q(z_<t|x)} q(z_t | z_<t,x) = q(z_t|x). This has less MC noise than
        # counting sampled z_t, while still marginalizing over entire histories.
        probabilities.append(q.reshape(num_samples, n, -1).mean(0))

    probs = torch.stack(probabilities, dim=1) * batch.mask[..., None]
    paths = torch.stack(states, dim=-1).reshape(num_samples, n, tmax)
    return {
        "probs": probs,
        "states": probs.argmax(-1).masked_fill(~batch.mask, -1),
        "samples": paths.masked_fill(~batch.mask[None], -1),
        "attention": alpha * batch.mask[..., None],
    }


@torch.no_grad()
def filter_prefixes(model, batch, num_samples=128):
    """Approximate q(z_t | x_0:t), evaluated independently at every prefix.

    Because the guide is amortized, truncating it is not guaranteed to produce
    exact p(z_t|x_0:t). Cost exceeds one smoothing pass; use posterior(prefix)
    and select its last visit when only the current online estimate is needed.
    """
    probs = [posterior(model, batch.prefix(t + 1), num_samples)["probs"][:, t] for t in range(batch.n_visits)]
    probs = torch.stack(probs, dim=1)
    return {"probs": probs, "states": probs.argmax(-1).masked_fill(~batch.mask, -1)}


def log_joint(model, batch, paths):
    """Compute log p_theta(x,z) for paths [S,N,T], without a Pyro trace.

    Using this mostly for diagnostics against the Pyro implementation and exact enumeration of states.
    """
    model.validate_batch(batch)
    samples, n, total = paths.shape

    if (n, total) != (batch.n_patients, batch.n_visits):
        raise ValueError("paths must have shape [S,N,T] matching the batch.")
    
    safe_paths    = paths.clamp(min=0)
    alpha         = model.attention(batch.context)
    emissions     = model.emissions.log_prob_all_states(batch.continuous, batch.binary)
    emissions     = emissions.unsqueeze(0).expand(samples, -1, -1, -1)
    emission_logp = emissions.gather(-1, safe_paths[..., None]).squeeze(-1)

    terms         = []
    
    for t in range(total):
        if t == 0:
            probs           = model.initial_probs.expand(samples, n, -1)
        else:
            transition_rows = model.transition_matrix[safe_paths[:, :, :t]]
            probs           = (alpha[None, :, t, :t, None] * transition_rows).sum(-2)
        terms.append(probs.gather(-1, safe_paths[:, :, t, None]).squeeze(-1).log())

    return ((torch.stack(terms, dim=-1) + emission_logp) * batch.mask[None]).sum(-1)


@torch.no_grad()
def exact_posterior(model, batch, max_paths=100_000):
    """Enumerate K**T paths for one unpadded short sequence."""
    
    if batch.n_patients != 1 or int(batch.lengths[0]) != batch.n_visits:
        raise ValueError("Exact diagnostic requires one unpadded trajectory.")
    
    k, total = model.config.num_states, batch.n_visits
    
    if k ** total > max_paths:
        raise ValueError(f"Would enumerate {k ** total:,} paths; pls shorten the sequence.")

    paths     = torch.tensor(list(product(range(k), repeat=total)), device=batch.lengths.device, dtype=torch.long)[:, None]
    logp      = log_joint(model, batch, paths)[:, 0]
    weights   = logp.softmax(0)
    marginals = (weights[:, None, None] * F.one_hot(paths[:, 0], k)).sum(0)
    
    return {"probs": marginals[None], "log_evidence": logp.logsumexp(0)}


@torch.no_grad()
def forecast(model, batch, horizon=3, num_samples=128):
    """Sample future paths/observations, integrating over q of the past.

    Returns arrays [S,N,H,...]. Initial uncertainty is from the variational
    posterior and future uncertainty comes from the fitted generative model.
    """
    if horizon < 1 or num_samples < 1:
        raise ValueError("horizon and num_samples must be positive.")
    if not torch.all(batch.lengths == batch.n_visits):
        raise ValueError("Forecast needs equal, unpadded history lengths; crop/group first.")

    history = posterior(model, batch, num_samples)["samples"]
    n, tmax = batch.n_patients, batch.n_visits
    states  = list(history.reshape(num_samples * n, tmax).unbind(-1))
    context = batch.context[None].expand(num_samples, -1, -1, -1).reshape(num_samples * n, tmax, -1)

    future_states, future_continuous, future_binary, future_probs = [], [], [], []
    
    for _ in range(horizon):
        alpha = model.attention.next_weights(context)
        probs = model.forward_probs(alpha, states)
        z     = dist.Categorical(probs=probs).sample()
        x     = model.emissions.continuous_distribution(z).sample()
        b     = model.emissions.binary_distribution(x, z).sample() if model.config.n_binary else x.new_empty(x.shape[0], 0)
        
        future_states.append(z.reshape(num_samples, n))
        future_continuous.append(x.reshape(num_samples, n, -1))
        future_binary.append(b.reshape(num_samples, n, model.config.n_binary))
        future_probs.append(probs.reshape(num_samples, n, -1).mean(0))
        states.append(z)

        # The sampled observation becomes part of the next step's history.
        context = torch.cat((context, torch.cat((x, b), dim=-1)[:, None]), dim=1)
    return {
        "states": torch.stack(future_states, dim=2),
        "continuous": torch.stack(future_continuous, dim=2),
        "binary": torch.stack(future_binary, dim=2),
        "state_probs": torch.stack(future_probs, dim=1),
    }
