"""Classify the 300-item sample with each local Ollama model and save raw predictions plus timing."""
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "sample_300.csv"
OUT_DIR = ROOT / "results"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODELS = [
    "qwen3.5:9b-q4_K_M",
    "gemma4:e4b-it-q4_K_M",
    "qwen3:30b-a3b-instruct-2507-q4_K_M",
]
LABELS = ("positive", "negative", "neutral")
PARALLEL = 4  # matches OLLAMA_NUM_PARALLEL on this machine

SYSTEM = (
    "You are a financial analyst. Classify the sentiment of a sentence from financial news "
    "from the point of view of an investor in the company mentioned. "
    'Answer with JSON only: {"label": "positive" | "negative" | "neutral"}.'
)
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(LABELS)}},
    "required": ["label"],
}


def parse_label(content: str) -> str | None:
    """Return a valid label, or None for anything unusable (counted as invalid output)."""
    try:
        label = str(json.loads(content).get("label", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError):
        return None
    return label if label in LABELS else None


async def classify(client: httpx.AsyncClient, sem: asyncio.Semaphore, model: str, row: pd.Series) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["text"]}],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
        "keep_alive": "10m",
    }
    # qwen3.5 and gemma4 both reason silently by default (gemma4 spent ~254 hidden tokens per item);
    # the task needs a label, not a chain of thought, so thinking is off for every model.
    body["think"] = False
    async with sem:
        try:
            r = await client.post(OLLAMA, json=body)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            return {"id": row["id"], "raw": f"ERROR: {exc}", "pred": None, "eval_count": 0, "eval_ns": 0}
    return {
        "id": row["id"],
        "raw": data["message"]["content"],
        "pred": parse_label(data["message"]["content"]),
        "eval_count": data.get("eval_count", 0),
        "eval_ns": data.get("eval_duration", 0),
    }


async def run_model(model: str, sample: pd.DataFrame) -> pd.DataFrame:
    sem = asyncio.Semaphore(PARALLEL)
    async with httpx.AsyncClient(timeout=300) as client:
        # Warm-up: load the model before the clock starts so load time doesn't skew items/minute
        await classify(client, asyncio.Semaphore(1), model, sample.iloc[0])
        t0 = time.perf_counter()
        rows = await asyncio.gather(*(classify(client, sem, model, row) for _, row in sample.iterrows()))
        wall = time.perf_counter() - t0
    out = pd.DataFrame(rows)
    out["model"] = model
    out["wall_seconds"] = wall
    return out


async def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    sample = pd.read_csv(SAMPLE)
    for model in sys.argv[1:] or MODELS:  # optional: pass model names to re-run a subset
        res = await run_model(model, sample)
        safe = model.replace(":", "_").replace("/", "_")
        res.to_csv(OUT_DIR / f"preds_{safe}.csv", index=False)
        print(f"{model}: {res['pred'].notna().sum()}/{len(res)} valid in {res['wall_seconds'].iloc[0]:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
