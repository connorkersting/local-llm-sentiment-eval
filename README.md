# Small local LLMs for bulk financial sentiment classification

How accurate and how fast are small open models at labelling financial-news sentiment on one
laptop GPU? On 300 sentences the best of three, qwen3.5:9b, scored 93.0% at 280 items a minute.
Always guessing "neutral" scores 63%. Claude Haiku 5.5 scored 90.3% with thinking off and 93.7%
with it on.

## Problem

Labelling thousands of short texts (news, tickets, reviews) with a frontier API costs money
and sends the data off-device. A small model running locally is free per item and private,
but is it accurate enough, and fast enough, to replace the API for bulk work?

## Approach

- **Data:** 300 sentences sampled with a fixed seed from Financial PhraseBank's
  all-annotators-agree subset: financial news labelled `positive` / `negative` / `neutral`
  from an investor's view (CC BY-NC-SA 3.0; see [SOURCE.md](SOURCE.md)). Mix: 189 neutral,
  77 positive, 34 negative.
- **Models** (all via [Ollama](https://ollama.com) 0.40.2, 4-bit `q4_K_M` quantization):
  - `qwen3.5:9b`: 9B dense
  - `gemma4:e4b`: ~8B total, ~4B effective
  - `qwen3:30b-a3b-instruct-2507`: 30B mixture-of-experts, ~3B active per token
- **Harness** ([run_eval.py](run_eval.py)): one fixed system prompt; `temperature 0`, `seed 42`;
  JSON-schema-constrained output (`{"label": ...}` with an enum); 4 requests in parallel
  (matches Ollama's `OLLAMA_NUM_PARALLEL=4`); the model is warmed up before timing.
  Anything that isn't one of the three labels counts as invalid and wrong.
- **Metrics** ([score.py](score.py)): accuracy, macro-F1 (unweighted mean of per-class F1;
  it punishes ignoring the minority classes), invalid-output rate, items/minute
  (wall clock, 300 items), generation tokens/s per request.
- **References:** a majority-class baseline (always `neutral`), and Claude (Anthropic) labelling
  items 0–99 blind as a frontier ceiling.
- **Hardware:** laptop with an RTX 3070 Laptop GPU (8 GB VRAM), 32 GB RAM, Windows 11;
  Ollama with flash attention and a q8_0 KV cache.

## Results

| System | Accuracy | Macro-F1 | Invalid | Items/min | Gen tok/s | Acc. on items 0–99 |
|---|---|---|---|---|---|---|
| Baseline: always `neutral` | 63.0% | 0.258 | 0% | n/a | n/a | 64.0% |
| **qwen3.5:9b** | **93.0%** | **0.928** | 0% | 280 | 67 | 94.0% |
| qwen3:30b-a3b (MoE) | 90.3% | 0.897 | 0% | 97 | 9 | 92.0% |
| gemma4:e4b | 89.3% | 0.891 | 0% | **526** | 45 | 89.0% |
| gemma4:e4b, thinking on (default) | 80.0% | 0.802 | 0% | 58 | 69 | 79.0% |
| Claude Haiku 5.5, Batch API, thinking off, effort low ($0.006 for 300 items) | 90.3% | 0.883 | 0% | n/a* | n/a | 90.0% |
| Claude Haiku 5.5, Batch API, adaptive thinking, effort medium ($0.006 for 300 items) | 93.7% | 0.921 | 0% | n/a* | n/a | 93.0% |
| Router: qwen3.5 + gemma4 agree → keep; disagree (19) → Haiku 5.5 (either variant) | 92.3% | 0.918 | 0% | n/a | n/a | n/a |
| Claude (frontier reference, 100 items) | n/a | n/a | 0% | n/a | n/a | **98.0%** (F1 0.980) |

\*Batch API turnaround was 124 s for 300 items, including queueing; that measures batch latency, not throughput.
Router numbers are replayed from saved predictions by [router.py](router.py).

**Cheap cloud vs local:** the small cloud model (Haiku 5.5, thinking off) scored below the local 9B model (90.3% vs
93.0%; the gap is 8 items, near the noise floor). It also did worse on the hard items: on the 19 sentences where the
two local models disagree, Haiku was right 68% of the time and qwen3.5 79%. So escalating disagreements to Haiku
lowered accuracy (92.3%). With thinking on, Haiku matched the local model overall (93.7% vs 93.0%, 2 items) but was
still 68% on those 19 items, so the router didn't improve. The disagreement items look genuinely ambiguous rather
than too hard for a small model. For this task the local model alone is the best value short of a frontier model.

**Prompt sensitivity:** the same qwen3.5 model run through the generic classify prompt in my
[local-worker](https://github.com/connorkersting/local-worker) tool scored 94.7% (284 of 300, macro-F1 0.941,
0 invalid, 78.5 s) on 2026-10-15. Its system prompt lists the three labels and adds one instruction: "Judge the
sentiment of this financial news sentence from an investor's point of view." That is 5 items more than this
harness's prompt, inside the noise for 300 items. The predictions are in
`results/local-worker/preds_2026-10-15.csv` (kept out of `results/preds_*.csv` so `score.py` doesn't pick them
up). Earlier runs of the tool used other instruction wording and saved no predictions, and they landed several
points above and below this, so treat prompt wording as worth several points either way. A fair prompt
comparison needs a held-out split so the prompt isn't tuned on the test set.

Gen tok/s is the per-request decode speed while 4 requests share the GPU. Items/min is the
number that matters for bulk work.

**Where the errors are:** every model's main mistake is calling a neutral, factual sentence
positive or negative, i.e. over-reading sentiment. Neutral items misread: qwen3.5 18/189,
MoE 26/189, gemma4 30/189 (thinking on: 59/189). Negatives are almost never missed (32–34 of 34).

## Which model wins for bulk classification, and why

**`qwen3.5:9b`, with thinking turned off.**
- It's the most accurate local model: 93% accuracy, macro-F1 0.93, against a 63% baseline.
- It runs at 280 items/minute, fully on the 8 GB GPU.
- It's within 4 points of the frontier reference on the shared 100 items (94% vs 98%).

**`gemma4:e4b` is the throughput pick:** about 1.9x faster (526 items/min) for about 4 points
less accuracy. It's a good fit for a first-pass triage where a second model checks low-confidence items.

**The 30B MoE isn't worth it on this hardware.** It needs 19 GB, so more than half of it runs
from system RAM on the CPU, and it's 3x slower than qwen3.5:9b while being less accurate. Mixture-of-experts saves
compute per token, not memory: the whole model still has to fit somewhere.

## Decisions (and why)

- **Thinking off for every model.** Both qwen3.5 and gemma4 reason silently by default. For gemma4
  it cost ~254 hidden tokens per item, made it **9x slower and 9 points less accurate**
  (it argued itself out of `neutral`). The thinking-on row is kept to show this. It came from an earlier
  version of `run_eval.py` that left thinking at the model default; the committed harness forces it off, so
  it can't regenerate that row.
- **JSON schema with an enum** instead of free text, so invalid outputs are structurally
  impossible to misparse (all models: 0% invalid).
- **All-agree subset**, so the gold labels are unambiguous and the score measures the model,
  not annotator noise. The trade-off: it's easier than real-world text.
- **Macro-F1 alongside accuracy**: with 63% neutral, a model can look decent on accuracy while
  ignoring the minority classes; the baseline's F1 of 0.26 shows the gap.
- **Fixed seeds and temperature 0** for repeatability. The sample is a simple random sample, not
  stratified, so it mirrors the source mix.

## Limitations

- 300 items: a 1-point accuracy gap is 3 items. The 95% interval at 93% is roughly ±3 points, so
  qwen3.5 vs the MoE vs gemma4 (93.0 / 90.3 / 89.3) are close; only the baseline and thinking-on
  gaps are clearly beyond noise.
- One prompt, no prompt tuning, no few-shot examples.
- The Claude reference is a single interactive labelling pass over 100 items, not an API run with
  this harness's prompt. Both of its misses were arguable `positive` items read as `neutral`.
- One machine; speed numbers depend on GPU, quantization and Ollama settings.

## Reproduce

`score.py`, `router.py` and `diagnose.py` replay everything from the saved predictions in `results/`, with
no GPU and no API key. Re-running the models needs Ollama and the three models pulled.

```bash
uv sync
uv run python sample.py       # needs data/raw from the source zip (see SOURCE.md)
uv run python run_eval.py     # needs Ollama with the three models pulled
uv run python run_eval_claude.py  # optional cloud row: needs ANTHROPIC_API_KEY in .env (see .env.example), ~$0.01
uv run python score.py        # writes results/summary.csv
uv run python router.py       # replays the escalation router from saved predictions
uv run python diagnose.py     # output lengths and confusion matrices
```

## Data and license

Financial PhraseBank © Malo et al. (2014), licensed
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). `data/sample_300.csv` is a random
300-sentence sample, re-indexed 0 to 299, and `data/claude_labels_100.csv` adds new labels to items 0 to 99.
Both are shared under the same license, for non-commercial use; see [SOURCE.md](SOURCE.md). The files in
`results/` hold item ids, model labels, token counts and costs, with no source sentences; since they label the
dataset, they are shared under the same CC BY-NC-SA 3.0 terms. The code is MIT ([LICENSE](LICENSE)).

## Who built what

Claude Code wrote the harness, the scoring and router scripts, and this README, in one session on 2026-10-11.
I set the question (can a laptop model replace a paid API for bulk labelling?) and the decision rules: thinking
off, a JSON-schema enum, a majority-class baseline and a frontier reference. The accuracy figures above were
recomputed from the committed prediction files on 2026-10-15.
