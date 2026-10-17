"""Pyro implementation of Attentive State-Space Models  
LSTM networks demonstrated in Figures 2 and 3 of our paper

Note that the attention output is a sequence of weights (i.e. this is NOT a Transformer or
ordinary dot-product attention over observation embeddings.) The weights here
will mix rows of a *single* baseline transition matrix in the latent space.
"""

import torch
from torch import nn


class StackedLSTMCells(nn.Module):
    """Expose hidden AND cell states at each prefix for the decoder bridge.

    nn.LSTM returns all final-layer outputs but only the final cell state. A
    short explicit loop lets us initialize a separate decoder at every prefix
    with the corresponding encoder states, including all layers.
    """

    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells      = nn.ModuleList([
            nn.LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(layers)
        ])

    def zero(self, batch_size, reference):
        shape = (len(self.cells), batch_size, self.hidden_dim)
        return reference.new_zeros(shape), reference.new_zeros(shape)

    def forward(self, inputs, state):

        old_h, old_c = state
        hs, cs       = [], []

        for i, cell in enumerate(self.cells):
            h, c   = cell(inputs, (old_h[i], old_c[i]))
            hs.append(h)
            cs.append(c)
            inputs = h
        return torch.stack(hs), torch.stack(cs)


class CausalSeq2SeqAttention(nn.Module):
    """Implementation for the encoder/decoder attention in Section 2.3.

    alpha[n,t,s] means influence of z_s on z_t, where s < t (zero-based). Here,
    the row for t=0 is zero because z_0 uses pi. All other rows sum to one, and
    the encoder for row t processes x_0,...,x_{t-1} but not x_t to implement the 
    transition model in eq (2).

    Decoder outputs are ordered newest-to-oldest and are then put back into
    chronological columns. (Note that compute and memory grow quadratically 
    with trajectory length.)
    """

    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.encoder = StackedLSTMCells(input_dim, hidden_dim, layers)
        self.decoder = StackedLSTMCells(hidden_dim, hidden_dim, layers)
        self.score   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        n, total, _        = x.shape
        if total == 1:
            return x.new_zeros(n, 1, 1)
        
        state              = self.encoder.zero(n, x)
        prefix_h, prefix_c = [], []

        for s in range(total - 1):
            state = self.encoder(x[:, s], state)
            prefix_h.append(state[0])
            prefix_c.append(state[1])

        # Each (patient, target time) pair has its own decoder and prefix state.
        h      = torch.stack(prefix_h, dim=2).reshape(len(self.encoder.cells), n * (total - 1), -1)
        c      = torch.stack(prefix_c, dim=2).reshape_as(h)
        inputs = h[-1]  # Last encoder output is the first decoder input.
        scores = []

        for lag in range(total - 1):
            h, c    = self.decoder(inputs, (h, c))
            inputs  = h[-1]  # Autoregressive feedback, as in Figure 2.
            scores.append(self.score(inputs).reshape(n, total - 1))

        scores        = torch.stack(scores, dim=-1)  # [N, target-1, lag]
        t             = torch.arange(1, total, device=x.device)
        lag           = torch.arange(total - 1, device=x.device)
        scores        = scores.masked_fill(lag[None, None, :] >= t[None, :, None], -torch.inf)
        lag_weights   = scores.softmax(-1)
        s             = torch.arange(total, device=x.device)
        index         = (t[:, None] - 1 - s[None, :]).clamp(min=0)
        chronological = lag_weights.gather(-1, index[None].expand(n, -1, -1))
        chronological = chronological * (s[None, :] < t[:, None])[None]
        
        return torch.cat((x.new_zeros(n, 1, total), chronological), dim=1)

    def next_weights(self, history):
        """Attention for the next unobserved state"""
        dummy = history.new_zeros(history.shape[0], 1, history.shape[-1])
        return self(torch.cat((history, dummy), dim=1))[:, -1, :-1]


class BackwardEncoder(nn.Module):
    """Summarize x_t:T using a reverse LSTM with right-padding excluded."""

    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.rnn = StackedLSTMCells(input_dim, hidden_dim, layers)

    def forward(self, x, mask):
        state   = self.rnn.zero(x.shape[0], x)
        outputs = []

        for t in reversed(range(x.shape[1])):
            new_h, new_c = self.rnn(x[:, t], state)
            active       = mask[:, t][None, :, None]
            # Ignoring padded steps prevents the backwards network from treating
            # artificial zeros after discharge as actual clinical observations.
            state        = (torch.where(active, new_h, state[0]), torch.where(active, new_c, state[1]))
            outputs.append(state[0][-1])
        return torch.stack(outputs[::-1], dim=1)


class AttentiveGuideNetwork(nn.Module):
    """Implements Equation (10) as softmax of a linear map of [backward h, forward probs]."""

    def __init__(self, input_dim, hidden_dim, layers, num_states):
        super().__init__()
        self.backward = BackwardEncoder(input_dim, hidden_dim, layers)
        self.combiner = nn.Linear(hidden_dim + num_states, num_states)

    def logits(self, backward_h, forward_probs):
        return self.combiner(torch.cat((backward_h, forward_probs), dim=-1))
