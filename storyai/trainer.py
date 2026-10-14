"""The training loop: show the model your stories and let it learn to predict them."""

import time
from pathlib import Path

import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .data import auto_vocab_size, encode_stories, get_batch, is_holdout, read_stories
from .model import GPT, PRESETS, GPTConfig
from .optim import AdamW, clip_grad_norm, learning_rate
from .tokenizer import END_OF_STORY, Tokenizer


def estimate_loss(model, data, batch_size, batches=8):
    rng = np.random.default_rng(1234)  # same batches every time, so numbers are comparable
    losses = [model.forward(*get_batch(data, batch_size, model.config.block_size, rng))[1]
              for _ in range(batches)]
    return float(np.mean(losses))


def preview(model, tokenizer, rng, tokens=48):
    eos = tokenizer.special_ids[END_OF_STORY]
    ids = list(model.generate([eos], tokens, temperature=0.8, stop_id=eos, rng=rng))
    return tokenizer.decode(ids).replace("\n", " ").strip()


def train(stories_dir, out_dir, size=None, steps=2000, batch_size=16, lr=1e-3,
          vocab_size=None, fresh=False, holdout=True, eval_every=100, seed=None):
    stories_dir, out_dir = Path(stories_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path, ckpt_path = out_dir / "tokenizer.json", out_dir / "model.npz"
    rng = np.random.default_rng(seed)

    stories = read_stories(stories_dir)
    if not stories:
        print(f"No stories found in {stories_dir}/")
        print("Put .txt files there, or run:  python story.py feed")
        return
    num_chars = sum(len(s) for s in stories)
    print(f"Found {len(stories)} {'story' if len(stories) == 1 else 'stories'} "
          f"({num_chars:,} characters).")

    if fresh:
        for path in (tok_path, ckpt_path):
            path.unlink(missing_ok=True)

    # 1. The tokenizer: learn how to chop text into tokens.
    if tok_path.exists():
        tokenizer = Tokenizer.load(tok_path)
        print(f"Using your tokenizer ({tokenizer.vocab_size} tokens).")
    else:
        vocab_size = vocab_size or auto_vocab_size(num_chars)
        print(f"Training the BPE tokenizer (target vocabulary: {vocab_size} tokens)...")
        tokenizer = Tokenizer.train(stories, vocab_size, verbose=True)
        tokenizer.save(tok_path)
        pieces = [tokenizer.decode([256 + i]) for i in range(len(tokenizer.merges))]
        words = [w for w in pieces if w.startswith(" ") and w[1:].isalpha()][-12:]
        print("  some tokens it invented:", " ".join(repr(w) for w in words))

    # 2. Split off a few stories as a quiz the model never studies.
    use_holdout = holdout and len(stories) >= 10
    train_stories = [s for s in stories if not (use_holdout and is_holdout(s))]
    val_stories = [s for s in stories if use_holdout and is_holdout(s)]
    if not train_stories or not val_stories:
        train_stories, val_stories = stories, []
    train_data = encode_stories(train_stories, tokenizer)
    val_data = encode_stories(val_stories, tokenizer) if val_stories else None
    print(f"Training on {len(train_stories)} stories ({len(train_data):,} tokens)", end="")
    if val_data is not None:
        print(f", quizzing on {len(val_stories)} held-out stories ({len(val_data):,} tokens).")
    else:
        print(". (No held-out quiz yet; one appears once you have about 10+ stories.)")

    # 3. The brain: start fresh, or keep teaching the one you already have.
    meta = {"steps_trained": 0}
    if ckpt_path.exists():
        model, optimizer, meta = load_checkpoint(ckpt_path)
        if model.config.vocab_size != tokenizer.vocab_size:
            raise SystemExit("The saved model and tokenizer don't match. Run with --fresh to start over.")
        model.rng = np.random.default_rng(seed)
        optimizer = optimizer or AdamW(model.params)
        print(f"Continuing to train your model ({meta.get('steps_trained', 0):,} steps so far).")
        if size:
            print(f"  (--size {size} ignored: an existing model keeps its size. Use --fresh to start over.)")
    else:
        size = size or "tiny"
        config = GPTConfig(vocab_size=tokenizer.vocab_size, **PRESETS[size])
        model = GPT(config, seed=rng.integers(1 << 31))
        optimizer = AdamW(model.params)
        print(f"Created a new '{size}' model with {model.num_params():,} parameters, all random.")
    c = model.config
    print(f"  {c.n_layer} layers, {c.n_head} heads, width {c.n_embd}, "
          f"reads {c.block_size} tokens of context.")

    warmup = min(100, max(1, steps // 10))
    best_val, best_step, checks_without_progress = float("inf"), 0, 0
    steps_at_start = meta.get("steps_trained", 0)

    def evaluate_and_save(step):
        nonlocal best_val, best_step, checks_without_progress
        train_loss = estimate_loss(model, train_data, batch_size)
        line = f"  [check] step {step}: train loss {train_loss:.3f}"
        meta["steps_trained"] = steps_at_start + step
        if val_data is None:
            save_checkpoint(ckpt_path, model, optimizer, meta)
            line += " (saved)"
        else:
            val_loss = estimate_loss(model, val_data, batch_size)
            line += f", quiz loss {val_loss:.3f}"
            if val_loss < best_val:
                best_val, best_step, checks_without_progress = val_loss, step, 0
                save_checkpoint(ckpt_path, model, optimizer, meta)
                line += " (best so far, saved)"
            else:
                checks_without_progress += 1
                line += f" (no better than step {best_step}, not saved)"
        print(line)
        if checks_without_progress == 3:
            print("  The quiz score has stopped improving, so it's memorising your stories now "
                  "rather than learning to write. Feeding it more stories helps; "
                  "you can also press Ctrl+C.")
        print(f"  it writes: {preview(model, tokenizer, rng)!r}")

    print(f"\nTraining for {steps} steps. Lower loss = better predictions. Ctrl+C stops safely.\n")
    step = 0
    train_time = 0.0
    try:
        evaluate_and_save(0)
        for step in range(1, steps + 1):
            started = time.time()
            x, y = get_batch(train_data, batch_size, c.block_size, rng)
            _, loss = model.forward(x, y, train=True)
            grads = model.backward()
            clip_grad_norm(grads, 1.0)
            lr_now = learning_rate(step - 1, steps, lr, warmup)
            optimizer.step(model.params, grads, lr_now)
            train_time += time.time() - started

            if step % 10 == 0 or step == 1:
                per_step = train_time / step
                left = per_step * (steps - step)
                print(f"step {step:>5}/{steps} | loss {loss:.3f} | lr {lr_now:.1e} | "
                      f"{per_step:.2f}s/step | ~{left / 60:.0f} min left", flush=True)
            if step % eval_every == 0 and step != steps:
                evaluate_and_save(step)
    except KeyboardInterrupt:
        print("\nStopping early...")
    evaluate_and_save(step)
    print(f"\nDone. Your model lives in {out_dir}/. Make it write with:  python story.py write")
