"""Pyro implementation of Attentive State-Space Models 
Observation-only initialization and Pyro SVI training.

For inference, the ELBO expectations and gradients are estimated via Monte Carlo.
"""

import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
import pyro
from pyro.infer import SVI, TraceGraph_ELBO
from pyro.optim import ClippedAdam


def initialize_from_observations(model, batch, config):
    """GMM emissions initialized with soft initial counts and pseudo-label guide pretraining.

    The mixture is fit to training visits
    only and never sees validation/test observations. The transition matrix is
    left at its generic initialization.
    """
    mask    = batch.mask
    x       = batch.continuous[mask].detach().cpu().numpy()
    b       = batch.binary[mask].detach().cpu().numpy()
    
    mixture = GaussianMixture(
        model.config.num_states, covariance_type="full", reg_covar=1e-3,
        n_init=5, random_state=config.seed
    ).fit(x)
    
    ref     = batch.continuous
    
    model.emissions.set_gaussian(
        torch.as_tensor(mixture.means_, dtype=ref.dtype, device=ref.device),
        torch.as_tensor(mixture.covariances_, dtype=ref.dtype, device=ref.device),
    )

    responsibilities = mixture.predict_proba(x)
    pseudo           = torch.zeros_like(mask, dtype=torch.long)
    pseudo[mask]     = torch.as_tensor(responsibilities.argmax(-1), device=ref.device)

    with torch.no_grad():
        initial_counts = F.one_hot(pseudo[:, 0], model.config.num_states).float().sum(0) + 1
        
        model.initial_logits.copy_(initial_counts.log())
        
        # Weighted logistic regressions initialize p(binary | continuous,state).
        # note that their weak L2 penalty is only an initialization not the SVI loss.
        for k in range(model.config.num_states):
            for j in range(model.config.n_binary):
                if len(np.unique(b[:, j])) == 2:
                    logistic = LogisticRegression(C=10.0, max_iter=300)
                    logistic.fit(x, b[:, j], sample_weight=responsibilities[:, k])
                    model.emissions.binary_weights[k, j].copy_(torch.as_tensor(logistic.coef_[0], dtype=ref.dtype, device=ref.device))
                    model.emissions.binary_bias[k, j].copy_(torch.as_tensor(logistic.intercept_[0], dtype=ref.dtype, device=ref.device))
                else:
                    p        = float(np.clip(b[:, j].mean(), 0.01, 0.99))
                    model.emissions.binary_bias[k, j].fill_(math.log(p / (1 - p)))

    # Initialize the backwards encoder and linear combiner. 
    optim  = torch.optim.Adam(model.guide_network.parameters(), lr=config.warm_start_lr)
    losses = []

    for step in range(config.warm_start_steps):
        indices = torch.randint(batch.n_patients, (min(config.batch_size, batch.n_patients),), device=ref.device)
        subset  = batch.select(indices)
        logits  = model.teacher_forced_logits(subset, pseudo[indices])
        loss    = F.cross_entropy(logits[subset.mask], pseudo[indices][subset.mask])

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.guide_network.parameters(), config.clip_norm)
        optim.step()
        losses.append(float(loss.detach()))

    return mixture, losses


def fit_svi(model, train, validation, config, param_state):
    """Optimize the loss in eq. (7) with unbiased categorical score-function gradients.

    Minibatches sample patients uniformly *with replacement*.

    Each SVI loss is the sum over its batch. 
    """
    model.train()
    batch_size      = min(config.batch_size, train.n_patients)
    steps_per_epoch = math.ceil(train.n_patients / batch_size)
    optimizer       = ClippedAdam({"lr": config.learning_rate, "clip_norm": config.clip_norm})
    elbo            = TraceGraph_ELBO(num_particles=config.num_particles)
    svi             = SVI(model.model, model.guide, optimizer, loss=elbo)
    
    history         = []
    start           = time.perf_counter()
    
    with pyro.get_param_store().scope(param_state):
        for epoch in range(1, config.epochs + 1):
            
            summed_loss, visits = 0.0, 0
            
            for _ in range(steps_per_epoch):
                
                idx    = torch.randint(train.n_patients, (batch_size,), device=train.lengths.device)
                subset = train.select(idx)
                loss   = svi.step(subset)
                
                if not math.isfinite(loss):
                    raise FloatingPointError("Nonfinite ELBO. Check scaling, covariance and learning rate.")
                summed_loss += loss
                visits += int(subset.lengths.sum())

            record = {"epoch": epoch, "train_negative_elbo": summed_loss / visits}
            
            if validation is not None:
                # This is an MC ELBO estimate. note that averaging over
                # particles does not turn it into a marginal-likelihood estimate.
                with torch.no_grad():
                    record["validation_negative_elbo"] = svi.evaluate_loss(validation) / int(validation.lengths.sum())
            history.append(record)
            
            if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
                extra = f", validation={record['validation_negative_elbo']:.3f}" if validation is not None else ""
                print(f"Epoch {epoch:3d}/{config.epochs}: negative ELBO/visit={record['train_negative_elbo']:.3f}{extra}", flush=True)

    return history, time.perf_counter() - start
