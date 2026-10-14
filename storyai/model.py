"""The brain: a GPT-style transformer written from scratch with NumPy.

Both directions are written out by hand: the forward pass (reading tokens and
predicting the next one) and the backward pass (backpropagation, which works
out how to nudge every weight to make better predictions). There is no deep
learning framework and no pretrained base model. Every weight starts out
random and is learned only from your stories.

Shape names used in comments:
  B = batch size, T = tokens in the context, N = B * T, C = n_embd (model width),
  V = vocab size, nh = number of attention heads, hs = C // nh (head size)
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128   # how many tokens the model can look back at
    n_layer: int = 4        # how many transformer blocks are stacked
    n_head: int = 4         # attention heads per block
    n_embd: int = 128       # size of the vector that represents each token
    dropout: float = 0.1    # randomly silences neurons while training to fight memorising


# Bigger brains write better but train more slowly.
PRESETS = {
    "tiny":   dict(n_layer=4, n_head=4, n_embd=128, block_size=128),
    "small":  dict(n_layer=6, n_head=6, n_embd=192, block_size=256),
    "medium": dict(n_layer=8, n_head=8, n_embd=256, block_size=256),
}

# Python floats (not NumPy scalars) so float32 arrays stay float32.
GELU_C = math.sqrt(2.0 / math.pi)


# --------------------------------------------------------------- building blocks
# Activations are 2D arrays with one row per token: (N, C) where N = B * T.
# (NumPy multiplies 2D matrices several times faster than 3D stacks.)
# Each *_forward returns its output plus what the matching *_backward needs.
# Many updates happen in place (x *= ...) because allocating big new arrays is slow.

def layernorm_forward(x, g, b, eps=1e-5):
    """Rescale each row to mean 0 and variance 1, then apply a learned scale and shift."""
    xhat = x - x.mean(1, keepdims=True)
    rstd = 1.0 / np.sqrt(np.mean(xhat * xhat, 1, keepdims=True) + eps)
    xhat *= rstd
    y = xhat * g
    y += b
    return y, (xhat, rstd, g)


def layernorm_backward(dy, cache):
    xhat, rstd, g = cache
    dg = (dy * xhat).sum(0)
    db = dy.sum(0)
    dxhat = dy * g
    dx = dxhat - dxhat.mean(1, keepdims=True)
    dx -= xhat * (dxhat * xhat).mean(1, keepdims=True)
    dx *= rstd
    return dx, dg, db


def linear_backward(dy, x, w):
    """Gradients of y = x @ w + b with respect to x, w and b."""
    return dy @ w.T, x.T @ dy, dy.sum(0)


def gelu_forward(x):
    """GELU activation, 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x^3))).
    Also returns the tanh part, which backward needs."""
    t = x * x
    t *= 0.044715
    t += 1.0
    t *= x
    t *= GELU_C
    np.tanh(t, out=t)
    y = t + 1.0
    y *= x
    y *= 0.5
    return y, t


def gelu_backward(dy, x, t):
    # d/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * GELU_C * (1 + 3 * 0.044715 * x^2)
    d = x * x
    d *= 3 * 0.044715
    d += 1.0
    d *= GELU_C
    d *= 1.0 - t * t
    d *= x
    d += 1.0 + t
    d *= 0.5
    d *= dy
    return d


def dropout_backward(dy, mask):
    if mask is None:
        return dy
    keep, scale = mask
    d = dy * keep
    d *= scale
    return d


# ------------------------------------------------------------------- the model

class GPT:
    def __init__(self, config, seed=0, dtype=np.float32):
        self.config = config
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)  # used for weight init and dropout
        self.params = self._init_params()
        self._cache = None
        # Added to attention scores: 0 where a token may look, -inf for future tokens.
        n = config.block_size
        self._mask = np.where(np.tril(np.ones((n, n), dtype=bool)), 0.0, -np.inf).astype(dtype)

    def _init_params(self):
        c, C = self.config, self.config.n_embd
        if C % c.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        std = 0.02
        proj_std = std / math.sqrt(2 * c.n_layer)  # keeps the residual stream calm in deep models

        def normal(*shape, s=std):
            return (self.rng.standard_normal(shape) * s).astype(self.dtype)

        def const(value, *shape):
            return np.full(shape, value, dtype=self.dtype)

        p = {
            "wte": normal(c.vocab_size, C),   # token embeddings (also used to predict tokens)
            "wpe": normal(c.block_size, C),   # position embeddings
        }
        for i in range(c.n_layer):
            h = f"h{i}."
            p[h + "ln1.g"], p[h + "ln1.b"] = const(1, C), const(0, C)
            p[h + "attn.w"], p[h + "attn.b"] = normal(C, 3 * C), const(0, 3 * C)
            p[h + "proj.w"], p[h + "proj.b"] = normal(C, C, s=proj_std), const(0, C)
            p[h + "ln2.g"], p[h + "ln2.b"] = const(1, C), const(0, C)
            p[h + "fc.w"], p[h + "fc.b"] = normal(C, 4 * C), const(0, 4 * C)
            p[h + "fc2.w"], p[h + "fc2.b"] = normal(4 * C, C, s=proj_std), const(0, C)
        p["lnf.g"], p["lnf.b"] = const(1, C), const(0, C)
        return p

    def num_params(self):
        return sum(v.size for v in self.params.values())

    def _dropout(self, x, p):
        """Zero a random fraction p of x (in place) and scale up the rest to compensate."""
        if p == 0.0:
            return x, None
        keep = self.rng.random(x.shape, dtype=np.float32) >= p
        scale = 1.0 / (1.0 - p)
        x *= keep
        x *= scale
        return x, (keep, scale)

    # ---------------------------------------------------------------- forward

    def forward(self, idx, targets=None, train=False):
        """Run the model on token ids `idx` of shape (B, T).

        With `targets` (B, T), returns (logits, loss) where loss is the average
        cross-entropy of predicting each next token. Without targets, returns
        (logits for the last position only, None), which is all generation needs.
        `train=True` turns on dropout and remembers what backward() needs.
        """
        c, p = self.config, self.params
        B, T = idx.shape
        if T > c.block_size:
            raise ValueError(f"context of {T} tokens is longer than block_size {c.block_size}")
        drop = c.dropout if train else 0.0

        x = (p["wte"][idx] + p["wpe"][:T]).reshape(B * T, c.n_embd)
        x, emb_mask = self._dropout(x, drop)
        block_caches = []
        for i in range(c.n_layer):
            x, cache = self._block_forward(x, B, T, i, drop)
            block_caches.append(cache)
        xf, lnf_cache = layernorm_forward(x, p["lnf.g"], p["lnf.b"])

        if targets is None:
            return xf.reshape(B, T, -1)[:, -1, :] @ p["wte"].T, None   # (B, V)

        # A score for every possible next token, at every position.
        logits = xf @ p["wte"].T                                      # (N, V)
        rows, flat_targets = np.arange(B * T), targets.reshape(-1)
        probs = logits - logits.max(1, keepdims=True)
        target_score = probs[rows, flat_targets]
        np.exp(probs, out=probs)
        sum_exp = probs.sum(1)
        loss = float(np.mean(np.log(sum_exp) - target_score))

        if train:
            probs /= sum_exp[:, None]
            self._cache = (idx, flat_targets, emb_mask, block_caches, xf, lnf_cache, probs)
        return logits.reshape(B, T, -1), loss

    def _block_forward(self, x, B, T, i, drop):
        p, h = self.params, f"h{i}."
        C = x.shape[1]
        nh = self.config.n_head
        hs = C // nh

        # Attention: every token gathers information from the tokens before it.
        a_in, ln1 = layernorm_forward(x, p[h + "ln1.g"], p[h + "ln1.b"])
        qkv = a_in @ p[h + "attn.w"]
        qkv += p[h + "attn.b"]                                  # (N, 3C)
        # Split into queries, keys and values, one (T, hs) matrix per sequence and head.
        q, k, v = np.ascontiguousarray(qkv.reshape(B, T, 3, nh, hs).transpose(2, 0, 3, 1, 4))
        att = q @ k.transpose(0, 1, 3, 2)                       # (B, nh, T, T) scores
        att *= 1.0 / math.sqrt(hs)
        att += self._mask[:T, :T]                               # no peeking at the future
        att -= att.max(-1, keepdims=True)                       # softmax
        np.exp(att, out=att)
        att /= att.sum(-1, keepdims=True)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B * T, C)
        a_out = y @ p[h + "proj.w"]
        a_out += p[h + "proj.b"]
        a_out, m1 = self._dropout(a_out, drop)
        x = x + a_out

        # MLP: each token thinks about what it has gathered.
        f_in, ln2 = layernorm_forward(x, p[h + "ln2.g"], p[h + "ln2.b"])
        f_pre = f_in @ p[h + "fc.w"]
        f_pre += p[h + "fc.b"]                                  # (N, 4C)
        f_act, f_tanh = gelu_forward(f_pre)
        f_out = f_act @ p[h + "fc2.w"]
        f_out += p[h + "fc2.b"]
        f_out, m2 = self._dropout(f_out, drop)
        x += f_out

        return x, (ln1, a_in, q, k, v, att, y, m1, ln2, f_in, f_pre, f_tanh, f_act, m2)

    # --------------------------------------------------------------- backward

    def backward(self):
        """Backpropagation: gradient of the last training loss for every parameter."""
        if self._cache is None:
            raise RuntimeError("call forward(..., targets, train=True) before backward()")
        idx, flat_targets, emb_mask, block_caches, xf, lnf_cache, probs = self._cache
        self._cache = None
        p, grads = self.params, {}
        B, T = idx.shape
        N = B * T

        # Cross-entropy + softmax: gradient is (predicted probs - one-hot target) / N.
        dlogits = probs
        dlogits[np.arange(N), flat_targets] -= 1.0
        dlogits *= 1.0 / N

        grads["wte"] = dlogits.T @ xf
        dx = dlogits @ p["wte"]
        dx, grads["lnf.g"], grads["lnf.b"] = layernorm_backward(dx, lnf_cache)
        for i in reversed(range(self.config.n_layer)):
            dx = self._block_backward(dx, B, T, i, block_caches[i], grads)
        dx = dropout_backward(dx, emb_mask)

        np.add.at(grads["wte"], idx.reshape(-1), dx)
        grads["wpe"] = np.zeros_like(p["wpe"])
        grads["wpe"][:T] = dx.reshape(B, T, -1).sum(0)
        return grads

    def _block_backward(self, dx, B, T, i, cache, grads):
        p, h = self.params, f"h{i}."
        ln1, a_in, q, k, v, att, y, m1, ln2, f_in, f_pre, f_tanh, f_act, m2 = cache
        C = dx.shape[1]
        nh = self.config.n_head
        hs = C // nh

        # MLP, in reverse.
        d = dropout_backward(dx, m2)
        d, grads[h + "fc2.w"], grads[h + "fc2.b"] = linear_backward(d, f_act, p[h + "fc2.w"])
        d = gelu_backward(d, f_pre, f_tanh)
        d, grads[h + "fc.w"], grads[h + "fc.b"] = linear_backward(d, f_in, p[h + "fc.w"])
        d, grads[h + "ln2.g"], grads[h + "ln2.b"] = layernorm_backward(d, ln2)
        dx = dx + d

        # Attention, in reverse.
        d = dropout_backward(dx, m1)
        d, grads[h + "proj.w"], grads[h + "proj.b"] = linear_backward(d, y, p[h + "proj.w"])
        d = np.ascontiguousarray(d.reshape(B, T, nh, hs).transpose(0, 2, 1, 3))  # (B, nh, T, hs)
        dscores = d @ v.transpose(0, 1, 3, 2)                   # gradient of att, for now
        dv = att.transpose(0, 1, 3, 2) @ d
        # Softmax backward. Masked positions have att == 0, so they get no gradient.
        dscores -= (dscores * att).sum(-1, keepdims=True)
        dscores *= att
        dscores *= 1.0 / math.sqrt(hs)
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q
        dqkv = np.stack([dq, dk, dv]).transpose(1, 3, 0, 2, 4).reshape(B * T, 3 * C)
        d, grads[h + "attn.w"], grads[h + "attn.b"] = linear_backward(dqkv, a_in, p[h + "attn.w"])
        d, grads[h + "ln1.g"], grads[h + "ln1.b"] = layernorm_backward(d, ln1)
        dx += d
        return dx

    # ------------------------------------------------------------- generation

    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=None, top_p=0.95,
                 stop_id=None, rng=None):
        """Continue the token list `ids`, yielding one new token id at a time."""
        rng = rng if rng is not None else np.random.default_rng()
        ids = list(ids)
        for _ in range(max_new_tokens):
            context = np.array([ids[-self.config.block_size:]])
            logits, _ = self.forward(context)
            next_id = sample_next(logits[0], temperature, top_k, top_p, rng)
            if next_id == stop_id:
                return
            ids.append(next_id)
            yield next_id


def sample_next(logits, temperature=0.8, top_k=None, top_p=0.95, rng=None):
    """Pick the next token from the model's scores.

    temperature: lower = safer and more predictable, higher = wilder. 0 = always the top pick.
    top_k: only consider the k most likely tokens.
    top_p: only consider the most likely tokens that together cover this much probability.
    """
    logits = logits.astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k and top_k < logits.size:
        cutoff = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < cutoff, -np.inf, logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        keep = (np.cumsum(sorted_probs) - sorted_probs) < top_p
        filtered = np.zeros_like(probs)
        filtered[order[keep]] = sorted_probs[keep]
        probs = filtered / filtered.sum()
    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.choice(probs.size, p=probs))
