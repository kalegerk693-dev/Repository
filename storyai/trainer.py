"""The training loop: show the model your stories and let it learn to predict them."""

import time
from pathlib import Path

import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .data import auto_vocab_size, encode_stories, get_batch, is_holdout, story_id, story_sources
from .model import GPT, PRESETS, GPTConfig
from .optim import AdamW, clip_grad_norm, learning_rate
from .originality import record_trained_stories
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


class TrainingError(Exception):
    """A problem the user has to fix (shown as a message, not a traceback)."""


def train(stories_dir, out_dir, size=None, steps=2000, batch_size=16, lr=1e-3,
          vocab_size=None, fresh=False, holdout=True, eval_every=100, seed=None,
          log=print, should_stop=None, on_progress=None):
    """Train the tokenizer (first time only) and the model on the stories in `stories_dir`.

    log: where progress messages go. should_stop: called every step; return True to
    stop early (Ctrl+C also works). on_progress: called with keyword details as
    training goes, for showing a progress bar.
    """
    should_stop = should_stop or (lambda: False)
    on_progress = on_progress or (lambda **info: None)
    stories_dir, out_dir = Path(stories_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path, ckpt_path = out_dir / "tokenizer.json", out_dir / "model.npz"
    rng = np.random.default_rng(seed)

    sources = story_sources(stories_dir)
    stories = [text for _, text in sources]
    if not stories:
        raise TrainingError("No stories to learn from yet. Add some stories first.")
    num_chars = sum(len(s) for s in stories)
    log(f"Found {len(stories)} {'story' if len(stories) == 1 else 'stories'} "
        f"({num_chars:,} characters).")

    if vocab_size is not None and vocab_size < 257:
        raise TrainingError("The vocabulary needs at least 257 tokens "
                            "(256 bytes plus the end-of-story token).")
    if fresh:
        for path in (tok_path, ckpt_path):
            path.unlink(missing_ok=True)
    if ckpt_path.exists() and not tok_path.exists():
        raise TrainingError("Found a trained model but its tokenizer.json is missing, so the two "
                            "can't be matched up. Start over (--fresh).")

    # 1. The tokenizer: learn how to chop text into tokens.
    if tok_path.exists():
        tokenizer = Tokenizer.load(tok_path)
        log(f"Using your tokenizer ({tokenizer.vocab_size} tokens).")
        if vocab_size:
            log("  (--vocab-size ignored: your tokenizer already exists. Start over (--fresh) to rebuild it.)")
        if tokenizer.trained_chars and num_chars > 4 * tokenizer.trained_chars:
            log("  Tip: you have far more text now than when the tokenizer was made. Starting over "
                "(--fresh) would build a better vocabulary, but the brain would relearn from zero.")
    else:
        vocab_size = vocab_size or auto_vocab_size(num_chars)
        log(f"Training the BPE tokenizer (target vocabulary: {vocab_size} tokens)...")
        tokenizer = Tokenizer.train(stories, vocab_size, verbose=True, log=log)
        tokenizer.save(tok_path)
        pieces = [tokenizer.decode([256 + i]) for i in range(len(tokenizer.merges))]
        words = [w for w in pieces if w.startswith(" ") and w[1:].isalpha()][-12:]
        log("  some tokens it invented: " + " ".join(repr(w) for w in words))

    # 2. The brain: start fresh, or keep teaching the one you already have.
    meta = {"steps_trained": 0}
    if ckpt_path.exists():
        model, optimizer, meta = load_checkpoint(ckpt_path)
        if model.config.vocab_size != tokenizer.vocab_size:
            raise TrainingError("The saved model and tokenizer don't match. Start over (--fresh).")
        model.rng = np.random.default_rng(seed)
        optimizer = optimizer or AdamW(model.params)
        log(f"Continuing to train your model ({meta.get('steps_trained', 0):,} steps so far).")
        if size:
            log(f"  (size '{size}' ignored: an existing model keeps its size. Start over (--fresh) to change it.)")
    else:
        size = size or "tiny"
        config = GPTConfig(vocab_size=tokenizer.vocab_size, **PRESETS[size])
        model = GPT(config, seed=rng.integers(1 << 31))
        optimizer = AdamW(model.params)
        log(f"Created a new '{size}' model with {model.num_params():,} parameters, all random.")
    c = model.config
    log(f"  {c.n_layer} layers, {c.n_head} heads, width {c.n_embd}, "
        f"reads {c.block_size} tokens of context.")

    # 3. Keep about 1 in 10 stories aside as a quiz the model never studies. Stories it
    # already studied in earlier runs can't be in the quiz, or the quiz would be too easy.
    seen = set(meta.get("seen", []))
    quiz = []
    if holdout and len(stories) >= 10:
        quiz = [s for s in stories if is_holdout(s) and story_id(s) not in seen]
        if quiz and sum(map(len, quiz)) > 0.3 * num_chars:
            log("(No quiz this run: the stories picked for it hold too much of your text.)")
            quiz = []
    in_quiz = set(quiz)
    train_stories = [s for s in stories if s not in in_quiz]
    train_data = encode_stories(train_stories, tokenizer)
    val_data = encode_stories(quiz, tokenizer) if quiz else None
    line = f"Training on {len(train_stories)} stories ({len(train_data):,} tokens)"
    if val_data is not None:
        line += f", quizzing on {len(quiz)} held-out stories ({len(val_data):,} tokens)."
    elif not holdout:
        line += ". (Quiz turned off with --no-holdout.)"
    elif len(stories) < 10:
        line += ". (No held-out quiz yet; one appears once you have about 10+ stories.)"
    else:
        line += ". (No quiz this run: none of the stories picked for it are new to the model.)"
    log(line)
    meta["seen"] = sorted(seen | {story_id(s) for s in train_stories})
    record_trained_stories(out_dir, [(name, text) for name, text in sources if text not in in_quiz])

    warmup = min(100, max(1, steps // 10))
    best_val, best_step, checks_without_progress = float("inf"), 0, 0
    steps_at_start = meta.get("steps_trained", 0)

    def save():
        try:
            save_checkpoint(ckpt_path, model, optimizer, meta)
            return True
        except OSError as e:  # e.g. the file is locked for a moment; keep training
            log(f"  (couldn't save this time: {e}. It will try again at the next check.)")
            return False

    def evaluate_and_save(step):
        nonlocal best_val, best_step, checks_without_progress
        meta["steps_trained"] = steps_at_start + step
        # Without a quiz, always keep the newest brain. Save first, before the slow checks.
        saved = val_data is None and save()
        train_loss = estimate_loss(model, train_data, batch_size)
        line = f"  [check] step {step}: train loss {train_loss:.3f}"
        if val_data is None:
            line += " (saved)" if saved else ""
        else:
            val_loss = estimate_loss(model, val_data, batch_size)
            line += f", quiz loss {val_loss:.3f}"
            if val_loss < best_val:
                if save():
                    best_val, best_step, checks_without_progress = val_loss, step, 0
                    line += " (best so far, saved)"
            else:
                checks_without_progress += 1
                line += f" (no better than step {best_step}, not saved)"
        log(line)
        if checks_without_progress == 3:
            log("  The quiz score has stopped improving, so it's memorising your stories now "
                "rather than learning to write. Feeding it more stories helps; "
                "you can also stop training here.")
        sample = preview(model, tokenizer, rng)
        log(f"  it writes: {sample!r}")
        on_progress(step=step, steps=steps, train_loss=train_loss, preview=sample,
                    val_loss=None if val_data is None else val_loss)

    log(f"\nTraining for {steps} steps. Lower loss = better predictions. Stopping early is safe.\n")
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
                log(f"step {step:>5}/{steps} | loss {loss:.3f} | lr {lr_now:.1e} | "
                    f"{per_step:.2f}s/step | ~{left / 60:.0f} min left")
                on_progress(step=step, steps=steps, loss=loss, seconds_left=left)
            if eval_every > 0 and step % eval_every == 0 and step != steps:
                evaluate_and_save(step)
            if should_stop():
                log("\nStopping early...")
                break
    except KeyboardInterrupt:
        log("\nStopping early... saving, please wait.")
    try:
        evaluate_and_save(step)
    except KeyboardInterrupt:  # impatient second Ctrl+C: still keep the work if we can
        if val_data is None:
            save()
        log("Stopped.")
        return
    log(f"\nDone. Your model is saved in {out_dir}/.")
