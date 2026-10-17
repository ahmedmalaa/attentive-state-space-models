# Method and implementation decisions

The source is the attached NeurIPS 2019 paper by Ahmed M. Alaa and Mihaela van der
Schaar. This is an independent Pyro implementation of its model and variational
architecture, not a line-by-line port of the authors' TensorFlow code.

## Equation-to-code map

| Paper component | Implementation | Details |
|---|---|---|
| Eq. (2), joint factorization | `AttentiveStateSpaceModel.model` | Per-visit categorical state and mixed emission; a plate over patients |
| Eq. (4), attentive transition | `forward_probs` | Sum of attention-weighted rows of one row-stochastic matrix P |
| Initial distribution | `initial_probs` | Softmax of a trainable K-vector |
| Eq. (5), attention A | `CausalSeq2SeqAttention` | LSTM prefix encoder, LSTM decoder initialized from that prefix, scalar logits and softmax over history |
| Eq. (6), observations | `MixedEmissions` | Multivariate Gaussian and conditionally independent logistic Bernoulli features given the continuous features and state |
| Eq. (7), ELBO | `fit_svi` | Pyro SVI with `TraceGraph_ELBO`, hard categorical samples and Adam |
| Eqs. (8)–(9), guide | `AttentiveStateSpaceModel.guide` | Samples whole autoregressive latent histories; shares A, P and pi with the model |
| Eq. (10), combiner | `AttentiveGuideNetwork.logits` | Linear map of concatenated backward LSTM hidden state and attentive forward probabilities |

State IDs are 0 through K-1 in code. An attention entry `alpha[n,t,s]` weights
the influence of past state `z[n,s]` on the distribution of `z[n,t]`. Row zero
contains no attention because the first state uses pi. Future and diagonal
entries are exactly zero. We mix *probabilities*, not logits.

## Causal indexing and decoder details

The joint distribution in Eq. (2) uses only past observations in the transition.
Eq. (5) and the encoder illustration instead use notation containing the current
visit. Feeding the current emission into its own state prior would generally
break this normalized ancestral factorization. We resolve the ambiguity by
encoding `x[:t]` when predicting `z[t]`. A unit test perturbs the current/future
observations and checks that the transition attention is unchanged.

The encoder exposes hidden and cell states at each prefix, for every layer.
These states initialize an autoregressive decoder. Its first input is the last
encoder output; later inputs are its preceding hidden output. A learned scalar
projection provides each historical position's score. The scores are normalized
over the valid history. For implementation clarity decoder positions are
newest-to-oldest, then rearranged into chronological columns. The paper does
not fully specify this output ordering or the output projection parameterization.

All target times can be decoded in parallel as a larger batch. There are still
O(T^2) attention entries; this does not turn the algorithm into a linear-time HMM.
The guide's backward LSTM explicitly skips right-padded visits, so padding cannot
be interpreted as future clinical evidence.

## Guide and gradient estimator

The inference network reads current/future observations through a backward LSTM
and summarizes sampled previous states through the shared attention and P.
The final combiner is linear before softmax, matching Eq. (10). This is an
approximation family; we do not assert that the summary is sufficient for an
arbitrary exact non-Markovian smoothing posterior.

A categorical sample is not reparameterizable. `TraceGraph_ELBO` constructs
the required score-function surrogate, using downstream dependency information
to reduce variance. We additionally enable Pyro's decaying-average baseline
at each state sample. The baseline is independent of the current sampled state
and changes gradient variance, not the ELBO's optimum. We do not use a
Gumbel-softmax relaxation, straight-through estimator, soft-state history, or
HMM-style forward-backward recursion.

The same `nn.Module` and transition parameters are registered in both Pyro
programs. Thus gradients from both the generative and variational terms update
the shared objects. Merely computing `log_p - log_q` from sampled integer states
and calling ordinary backpropagation would miss the score-function term.

Generic enumeration is exponential because every past state can influence a
future state. `pyro.markov` would assert a dependency structure that is not
present. The exact diagnostic enumerates all K^T paths explicitly, with a
maximum-path guard. It is only intended for very short single sequences.

## Emissions and estimation

Continuous observations follow a state-specific full-covariance Gaussian by
default. A positive Cholesky diagonal uses `softplus(raw_scale) + min_scale`;
strict lower-triangular entries are unconstrained. This parameterization ensures
a positive-definite covariance and bounds the Cholesky diagonal away from zero
(it is not an eigenvalue lower bound). The `diag` option is an explicit restricted
variant. Binary features have state-specific intercepts and coefficients on the
current continuous observation, so they are not simply independent Bernoullis
conditioned only on state.

The paper describes separate emission MLE updates. Here all trainable parameters
are updated with Adam on the same Monte Carlo ELBO, a generalized variational
optimization approach. There are no parameter priors or posterior distributions
over weights. This optimizes a variational approximation to maximum likelihood,
not a fully Bayesian parameter posterior. The configured minimum Gaussian scale
restricts the parameter space and limits covariance collapse.

## Additional initialization used in the demonstration

1. Fit a full-covariance Gaussian mixture to **training biomarkers only**.
2. Initialize Gaussian parameters from the mixture. Initialize binary logistic
   regressions using mixture responsibilities as weights.
3. Initialize pi from first-visit mixture assignments plus a small pseudocount.
   Leave P at its generic diagonal-favoring initialization; adjacent mixture-state
   counts are not the correct attentive-transition MLE.
4. Pretrain only the backward LSTM and combiner on mixture pseudo-labels.
5. Remove the pseudo-label objective and optimize the ELBO with sampled states.

No generator parameters or true latent labels enter these steps. This warm start
is an explicit practical extension to improve discrete-latent optimization. It
can be disabled. The notebook scores both the GMM and initialized guide to avoid
attributing all recovery to the later SVI steps or to attention.

## Differences from the paper's empirical configuration

| Choice | This package |
|---|---|
| Framework | PyTorch and Pyro instead of TensorFlow |
| Attention indexing | Strictly past-only, resolving ambiguous paper notation |
| Demo network size | One 32-unit layer; configure two 100-unit layers to match the paper's stated size |
| Optimizer | Adam on all parameters; no separate emission MLE loop |
| Discrete gradients | Explicit Pyro score-function estimator with baseline |
| Initialization | Optional observation-only GMM and guide warm start |
| Data | Transparent synthetic non-Markovian generator, not UK CF registry |
| Data support | Fully observed mixed features with variable-length right-padding |

## Evaluation and interpretation

The split is by patient, not by visit. Only training observations initialize or
fit parameters. Validation losses are monitored without selecting an epoch;
test patients are evaluated after the specified training run. A permutation
matching learned and true synthetic emission means is used for reporting. It
does not change the trained model. ARI is inherently permutation-invariant.

The returned posterior probabilities are marginals of the learned guide,
estimated by Monte Carlo over histories. They are not exact posterior marginals.
Taking their argmax yields marginal modes, not a globally optimal latent path.
Smoothing can use future supplied visits; prefix filtering cannot. Forecasting
starts from a sampled guide history and ancestrally generates new states and
observations using past-only attention.

State recovery is not proof of attention identifiability. Many attention maps
can produce similar transition mixtures when historical states repeat. P and
attention may also compensate for each other. The notebook shows true/learned
attention and the transition error separately, without claiming exact recovery.
It does not infer clinical causal mechanisms from these associative weights.

Missing feature values, irregular elapsed times, treatment interventions,
monotonic progression constraints and Bayesian parameter uncertainty are not
included. The number of states is supplied by the user. A clinical or benchmark
study would require appropriate data handling, model selection, multiple seeds
and uncertainty/calibration assessment beyond this synthetic demonstration.
