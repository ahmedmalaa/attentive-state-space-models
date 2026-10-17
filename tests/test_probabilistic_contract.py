"""Pyro implementation of Attentive State-Space Models"""

import torch
import pyro
import pytest
from assm import AttentiveSSM, ModelConfig, TrainingConfig, SequenceBatch
from assm.inference import log_joint
from pyro.infer import TraceGraph_ELBO


@pytest.fixture(autouse=True)
def reproducible():
    torch.set_num_threads(1)
    pyro.set_rng_seed(101)


def tiny_batch():
    return SequenceBatch(torch.randn(3, 5, 2), torch.randint(0, 2, (3, 5, 1)).float(), torch.tensor([5, 3, 1]))


def small_model(**kwargs):
    return AttentiveSSM(ModelConfig(hidden_dim=8, **kwargs))


def test_attention_is_causal_normalized_and_uses_older_history():
    model, batch = small_model(), tiny_batch()
    a = model.attention_weights(batch)
    assert torch.all(a.triu() == 0)
    assert torch.allclose(a[0, 1:].sum(-1), torch.ones(4))
    assert a[0, -1, :-2].sum() > 0
    changed = batch.context.clone()
    changed[:, 3:] += 1000
    other = model.model.attention(changed)
    assert torch.allclose(a[0, :4], other[0, :4])


def test_hmm_limit_and_convex_transition_mixture():
    model    = small_model().model
    previous = [torch.tensor([0, 2]), torch.tensor([1, 0])]
    a        = torch.tensor([[0., 1.], [0., 1.]])
    p        = model.forward_probs(a, previous)

    assert torch.allclose(p, model.transition_matrix[previous[-1]])

    a        = torch.tensor([[0.25, 0.75], [0.25, 0.75]])
    expected = 0.25 * model.transition_matrix[previous[0]] + 0.75 * model.transition_matrix[previous[1]]
    
    assert torch.allclose(model.forward_probs(a, previous), expected)


def test_pyro_joint_matches_manual_factorization_with_padding():
    estimator, batch = small_model(), tiny_batch()
    paths = torch.tensor([[0, 1, 2, 1, 0], [2, 2, 0, 0, 0], [1, 0, 0, 0, 0]])
    conditioned = pyro.poutine.condition(estimator.model.model, data={f"z_{t}": paths[:, t] for t in range(5)})
    with pyro.get_param_store().scope():
        trace = pyro.poutine.trace(conditioned).get_trace(batch)
        pyro_logp = trace.log_prob_sum()
    manual_logp = log_joint(estimator.model, batch, paths[None]).sum()
    assert torch.allclose(pyro_logp, manual_logp, atol=1e-5)


def test_backward_encoder_ignores_padding_and_single_visit_works():
    estimator, batch = small_model(), tiny_batch()
    h = estimator.model.guide_network.backward(batch.context, batch.mask)
    short = batch.select([1]).prefix(3)
    h_short = estimator.model.guide_network.backward(short.context, short.mask)
    assert torch.allclose(h[1, :3], h_short[0], atol=1e-6)
    one = estimator.posterior(batch.prefix(1), num_samples=3)
    assert one['probs'].shape == (3, 1, 3)
    assert torch.allclose(one['probs'].sum(-1), torch.ones(3, 1))
    assert torch.count_nonzero(one['attention']) == 0


def test_score_function_svi_reaches_model_and_guide_parameters():
    model, batch = small_model().model, tiny_batch()
    with pyro.get_param_store().scope():
        loss = TraceGraph_ELBO(num_particles=2).loss_and_grads(model.model, model.guide, batch)
        assert torch.isfinite(torch.tensor(loss))
        for module in [model.attention, model.guide_network, model.emissions]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(float(g.abs().sum()) for g in grads) > 0
        assert model.transition_logits.grad.abs().sum() > 0


def test_exact_enumeration_normalizes_and_is_guarded():
    estimator, batch = small_model(), tiny_batch().select([0]).prefix(3)
    result = estimator.exact_posterior(batch)
    assert torch.allclose(result['probs'].sum(-1), torch.ones(1, 3), atol=1e-6)
    assert torch.isfinite(result['log_evidence'])
    with pytest.raises(ValueError, match='enumerate'):
        estimator.exact_posterior(batch, max_paths=2)


def test_checkpoint_roundtrip_and_forecast_shapes(tmp_path):
    estimator, batch = small_model(), tiny_batch().prefix(1)
    estimator.save(tmp_path / 'model.pt')
    restored = AttentiveSSM.load(tmp_path / 'model.pt')
    pyro.set_rng_seed(8)
    first = estimator.posterior(batch, num_samples=5)
    pyro.set_rng_seed(8)
    second = restored.posterior(batch, num_samples=5)
    assert torch.equal(first['probs'], second['probs'])
    f = restored.forecast(batch, horizon=2, num_samples=4)
    assert f['continuous'].shape == (4, 3, 2, 2)
    assert f['states'].shape == (4, 3, 2)
    assert torch.allclose(f['state_probs'].sum(-1), torch.ones(3, 2), atol=1e-6)
    with pytest.raises(ValueError, match='equal'):
        restored.forecast(tiny_batch())


def test_continuous_only_diagonal_model_and_validation():
    estimator = small_model(n_binary=0, covariance='diag')
    batch = SequenceBatch(torch.randn(2, 3, 2), torch.empty(2, 3, 0), torch.tensor([3, 3]))
    with pyro.get_param_store().scope():
        assert torch.isfinite(torch.tensor(TraceGraph_ELBO().loss_and_grads(estimator.model.model, estimator.model.guide, batch)))
    assert estimator.forecast(batch, 1, 2)['binary'].shape == (2, 2, 1, 0)
    with pytest.raises(ValueError, match='finite'):
        SequenceBatch(torch.full((1, 1, 2), float('nan')), torch.zeros(1, 1, 1), torch.tensor([1]))


def test_fit_isolated_store_and_nondivisible_batch():

    batch = tiny_batch()

    with pyro.get_param_store().scope():

        sentinel  = pyro.param('external_parameter', torch.tensor(2.0))
        estimator = small_model()
        estimator.fit(batch, training=TrainingConfig(epochs=1, batch_size=2, num_particles=1, warm_start=False))
        assert set(pyro.get_param_store().keys()) == {'external_parameter'}
        assert pyro.param('external_parameter') is sentinel


def test_filter_cannot_see_future_visits():

    estimator, batch = small_model(), tiny_batch()
    x, b             = batch.continuous.clone(), batch.binary.clone()
    x[:, 3:]        += 50
    changed          = SequenceBatch(x, b, batch.lengths)

    pyro.set_rng_seed(900)

    p                = estimator.filter(batch, num_samples=8)['probs']

    pyro.set_rng_seed(900)

    q                = estimator.filter(changed, num_samples=8)['probs']

    assert torch.equal(p[:, :3], q[:, :3])
