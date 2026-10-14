# Story AI, built from scratch

A story-writing AI where **every part is ours**:

- **Our own tokenizer.** A byte-level BPE (byte pair encoding) tokenizer that learns its vocabulary from your stories.
- **Our own brain.** A GPT-style transformer with attention, MLPs, layer norm and embeddings, written line by line.
- **Our own learning.** Backpropagation is worked out by hand for every layer, plus our own AdamW optimizer and training loop.
- **No base model.** It starts as random numbers. Everything it knows comes from the stories **you** feed it.

The only library it uses is NumPy, for basic arrays and matrix multiplication. There is no PyTorch, no TensorFlow, no Hugging Face, and no pretrained weights.

## Use it in one click

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/kalegerk693-dev/Repository?quickstart=1)

Press the button and then **Create codespace**. After a minute or two, the Story AI app opens in a new browser tab. If your browser blocks the tab, open the **Ports** tab at the bottom of the editor and click the 🌐 globe icon next to *Story AI*. Everything is saved in your codespace, so you can come back to it from [github.com/codespaces](https://github.com/codespaces).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kalegerk693-dev/Repository/blob/master/StoryAI.ipynb)

Or use Google Colab: press ▶ on the first cell, then click the link it shows. Colab forgets everything when you close it, so use the notebook's download cell to save your AI.

The app has three steps, each with buttons:

1. **Feed it stories.** Paste stories or upload `.txt` files.
2. **Teach it.** Press *Start training* and watch the loss drop and its writing improve.
3. **Let it write.** Give it an opening line (or nothing) and press *Write a story*.

## On your own computer

```bash
pip install numpy
python story.py app        # then open http://localhost:8000
```

Or use the commands directly:

```bash
python story.py feed                       # paste a story (or drop .txt files into stories/)
python story.py train                      # teach it
python story.py write "Once upon a time"   # let it write an original story
```

## 1. Feeding it stories

The `stories/` folder is the AI's whole world. There are three ways to put stories in:

- Save each story as its own `.txt` file in `stories/`.
- Put many stories in one `.txt` file, separated by a line containing only `<|endofstory|>`.
- Run `python story.py feed`, paste or type a story, then type `END` on its own line.
  `python story.py feed --file my_story.txt` copies a file in for you.

**How much do you need?** More stories make it more original:

| What you feed it | What to expect |
|---|---|
| A handful of stories | It mostly memorises and repeats them |
| ~100 KB of text (~20 short stories) | Real-looking words and sentences that often don't make sense |
| 1 MB+ (hundreds of stories) | Its own stories, with characters, dialogue and events you never wrote. Longer training and a bigger brain make them hang together better |

## 2. Training

```bash
python story.py train                 # tiny brain, 2000 steps
python story.py train --size small    # bigger brain: writes better, trains slower
python story.py train --steps 5000    # train longer
```

The first time you train, it:

1. **Builds its tokenizer.** BPE reads all your stories and invents tokens for common pieces of text, like `' dragon'` or `' Once'`. The vocabulary size is picked automatically from how much text you have.
2. **Creates a brain** full of random numbers.
3. **Learns.** It reads random snippets of your stories, guesses each next token, measures how wrong it was (the *loss*), and backpropagation adjusts every weight to be a little less wrong next time.

Here's real output from training the tiny brain on 120 public-domain fairy tales (1.3 MB):

```
  [check] step 0: train loss 7.772, quiz loss 7.772 (best so far, saved)
  it writes: 'GZ! liftedancing trees fellowcond lion grass grandmotherFlledsel coat pieceach...'
  ...
step  1000/1500 | loss 4.147 | lr 3.6e-04 | 0.30s/step | ~3 min left
  ...
  [check] step 1250: train loss 4.087, quiz loss 4.661 (best so far, saved)
  it writes: 'It is lirm! there," said the Mar, 'But I will be able to a coat.' Then the evening
              the door, and the little dog, and sat down to the wolf...'
```

- **loss**: lower means it predicts your writing better. It starts around 7–8 (random guessing).
- **quiz loss**: about 1 in 10 stories (once you have 10+) is kept out of training as a quiz. If the quiz loss stops improving while the train loss keeps dropping, the AI is memorising rather than learning to write. Only the best quiz score gets saved, so you keep the version that writes most originally. Add `--no-holdout` to train on every story.
- **it writes**: a short live sample, so you can watch it go from gibberish to stories.

**Ctrl+C stops safely** at any time, and progress is kept.

**Adding more stories later:** feed them in and run `python story.py train` again. It picks up where it left off and keeps learning, now including the new stories. To throw everything away and start from zero (for example after adding lots of new text, so the tokenizer can relearn its vocabulary), use `--fresh`.

### Brain sizes

| `--size` | Parameters | Layers × width | Context | Speed (4-core CPU) |
|---|---|---|---|---|
| `tiny` (default) | ~1M | 4 × 128 | 128 tokens | ~0.3 s/step |
| `small` | ~3M | 6 × 192 | 256 tokens | ~1.5 s/step |
| `medium` | ~7M | 8 × 256 | 256 tokens | ~3.7 s/step |

Other options: `--lr` (learning rate), `--batch-size`, `--vocab-size`, `--seed`. See `python story.py train --help`.

## 3. Writing stories

```bash
python story.py write                              # a completely new story
python story.py write "The dragon was afraid of"   # you give the opening
python story.py write --count 3                    # three different stories
python story.py write --temperature 0.6            # safer, more predictable
python story.py write --temperature 1.1            # wilder, more surprising
python story.py write --max-tokens 1500            # longer stories
```

The story streams out token by token. It stops when the AI decides the story is over: it learned *how stories end* from yours.

Curious how the tokenizer sees text? Try:

```bash
python story.py tokens "Once upon a time, a dragon sneezed."
# [On][ce][ upon][ a][ time][,][ a][ drag][on][ s][ne][e][z][ed][.]
```

(That's the tokenizer trained on the fairy tales. Yours will split text differently, depending on which words your stories use most.)

## How it works

| File | What it does |
|---|---|
| `storyai/tokenizer.py` | Byte-level BPE: training (with fast incremental pair counting), encoding, decoding |
| `storyai/model.py` | The transformer: forward pass, hand-written backpropagation, and sampling |
| `storyai/optim.py` | AdamW optimizer, gradient clipping, learning-rate schedule |
| `storyai/data.py` | Reads your stories, turns them into a token stream, makes training batches |
| `storyai/trainer.py` | The training loop: quiz split, checks, saving the best model |
| `storyai/checkpoint.py` | Saves and loads the brain (`checkpoints/model.npz`) |
| `storyai/writer.py` | Loads a trained brain and writes a story piece by piece |
| `storyai/cli.py` | The `app` / `feed` / `train` / `write` / `tokens` commands |
| `storyai/webapp.py`, `storyai/app.html` | The point-and-click web app (Python's built-in web server, nothing to install) |
| `.devcontainer/`, `StoryAI.ipynb` | One-click setups for GitHub Codespaces and Google Colab |

The brain, step by step:

1. Each token gets a learned **embedding** (a vector of numbers), plus a vector for its **position**.
2. Each of the N **transformer blocks** does:
   - **Self-attention**: every token looks back at earlier tokens and gathers what matters. A causal mask stops it from seeing the future.
   - An **MLP**: each token "thinks" about what it gathered.
   - Both parts use layer norm and residual connections.
3. The final vectors are compared against every token embedding, which gives a score for each possible next token.
4. **Training** uses cross-entropy loss and backpropagation through all of the above.
5. **Writing** samples one token at a time from those scores, using temperature and top-p, and feeds each new token back in.

## Tests

```bash
python -m unittest discover -s tests -t .
```

These include a **gradient check**: every hand-written backward pass is compared against numerical derivatives, which proves the backpropagation math is right. There are also tokenizer tests, including one that checks the fast BPE trainer learns exactly the same merges as the plain, slow algorithm, plus an end-to-end feed → train → write test.
