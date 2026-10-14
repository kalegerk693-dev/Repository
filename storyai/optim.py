"""AdamW, the optimizer that turns gradients into weight updates."""

import math

import numpy as np


class AdamW:
    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}  # running average of gradients
        self.v = {k: np.zeros_like(v) for k, v in params.items()}  # running average of squared gradients
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        b1, b2 = self.betas
        bias1 = 1.0 - b1 ** self.t
        bias2 = 1.0 - b2 ** self.t
        for k, p in params.items():
            g, m, v = grads[k], self.m[k], self.v[k]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            if p.ndim >= 2 and self.weight_decay:
                p *= 1.0 - lr * self.weight_decay  # gently shrink weight matrices
            p -= lr * (m / bias1) / (np.sqrt(v / bias2) + self.eps)


def clip_grad_norm(grads, max_norm):
    """Scale all gradients down if together they are too large. Returns the original norm."""
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for g in grads.values():
            g *= scale
    return norm


def learning_rate(step, total_steps, max_lr, warmup):
    """Warm up linearly, then glide down to 10% along a cosine curve."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return max_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))
