import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from storyai.checkpoint import load_checkpoint, save_checkpoint
from storyai.model import GPT, GPTConfig
from storyai.optim import AdamW


def tiny_model(dropout=0.0, dtype=np.float64):
    cfg = GPTConfig(vocab_size=13, block_size=8, n_layer=2, n_head=2, n_embd=8, dropout=dropout)
    model = GPT(cfg, seed=1, dtype=dtype)
    rng = np.random.default_rng(2)
    # Push every parameter away from its tidy initial value so no gradient is trivially zero.
    for v in model.params.values():
        v += rng.normal(0, 0.1, v.shape)
    return model


class GradientCheck(unittest.TestCase):
    """Compare hand-written backprop against numerical finite differences."""

    def check(self, dropout):
        model = tiny_model(dropout)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, 13, (3, 7))
        targets = rng.integers(0, 13, (3, 7))

        def loss():
            model.rng = np.random.default_rng(3)  # identical dropout masks every call
            return model.forward(idx, targets, train=True)[1]

        loss()
        grads = model.backward()
        self.assertEqual(set(grads), set(model.params))
        eps = 1e-5
        for name, p in model.params.items():
            flat = p.reshape(-1)
            for j in rng.choice(p.size, min(p.size, 10), replace=False):
                old = flat[j]
                flat[j] = old + eps
                up = loss()
                flat[j] = old - eps
                down = loss()
                flat[j] = old
                numeric = (up - down) / (2 * eps)
                analytic = grads[name].reshape(-1)[j]
                self.assertLess(abs(numeric - analytic), 1e-7 + 1e-5 * abs(numeric),
                                f"{name}[{j}]: numeric {numeric} vs backprop {analytic}")

    def test_gradients(self):
        self.check(dropout=0.0)

    def test_gradients_with_dropout(self):
        self.check(dropout=0.3)


class ModelBehaviour(unittest.TestCase):
    def test_initial_loss_is_near_uniform_guess(self):
        cfg = GPTConfig(vocab_size=500, block_size=16, n_layer=2, n_head=2, n_embd=32)
        model = GPT(cfg)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, 500, (4, 16))
        logits, loss = model.forward(idx, rng.integers(0, 500, (4, 16)))
        self.assertEqual(logits.dtype, np.float32)
        self.assertAlmostEqual(loss, math.log(500), delta=0.1)

    def test_future_tokens_do_not_affect_the_past(self):
        model = tiny_model()
        idx = np.array([[1, 2, 3, 4, 5]])
        before = model.forward(idx[:, :3])[0]
        changed = idx.copy()
        changed[0, 3:] = [9, 9]
        logits_full, _ = model.forward(changed, np.zeros_like(changed))
        np.testing.assert_allclose(before[0], logits_full[0, 2], rtol=1e-10)

    def test_learns_a_repeating_pattern(self):
        cfg = GPTConfig(vocab_size=10, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
        model = GPT(cfg, seed=0)
        opt = AdamW(model.params, weight_decay=0.0)
        seq = np.tile(np.array([1, 5, 2, 7, 3]), 10)
        x, y = seq[None, :16], seq[None, 1:17]
        for _ in range(150):
            model.forward(x, y, train=True)
            opt.step(model.params, model.backward(), lr=3e-3)
        self.assertLess(model.forward(x, y)[1], 0.05)

    def test_checkpoint_roundtrip(self):
        cfg = GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=8)
        model = GPT(cfg, seed=4)
        opt = AdamW(model.params)
        idx = np.array([[1, 2, 3]])
        model.forward(idx, idx, train=True)
        opt.step(model.params, model.backward(), lr=1e-3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.npz"
            save_checkpoint(path, model, opt, {"steps_trained": 1})
            loaded, loaded_opt, meta = load_checkpoint(path)
        self.assertEqual(meta, {"steps_trained": 1})
        self.assertEqual(loaded.config, cfg)
        self.assertEqual(loaded_opt.t, 1)
        np.testing.assert_array_equal(model.forward(idx)[0], loaded.forward(idx)[0])
        np.testing.assert_array_equal(opt.m["wte"], loaded_opt.m["wte"])


if __name__ == "__main__":
    unittest.main()
