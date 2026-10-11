"""Classify the same 300-item sample with Claude Haiku 5.5 through the Message Batches API.

Cheap-cloud reference row for the local models: same system prompt, same JSON-schema enum, thinking off
(mirrors think=False on the local runs), lowest effort. Billed to the API key in .env (never printed).
"""
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from run_eval import SAMPLE, SCHEMA, SYSTEM, parse_label

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "results"
MODEL = "claude-haiku-5-5"
# Batch API list prices for Haiku 5.5, prompts up to 100K tokens (USD per million tokens), checked 2026-10-11:
# https://platform.claude.com/docs/en/about-claude/pricing
BATCH_IN_PER_MTOK = 0.05
BATCH_OUT_PER_MTOK = 0.25
POLL_SECONDS = 15

# Structured outputs need additionalProperties: false; otherwise the schema matches the local runs exactly.
CLAUDE_SCHEMA = {**SCHEMA, "additionalProperties": False}


# "off" mirrors think=False on the local runs. "adaptive" lets Haiku reason at its default effort; thinking
# tokens count toward max_tokens, so it gets more room.
VARIANTS: dict[str, dict[str, object]] = {
    "off": {"label": "thinking off", "thinking": {"type": "disabled"}, "effort": "low", "max_tokens": 256},
    "adaptive": {"label": "adaptive thinking, effort medium", "thinking": {"type": "adaptive"},
                 "effort": "medium", "max_tokens": 4096},
}


def build_requests(sample: pd.DataFrame, variant: str = "off") -> list[Request]:
    v = VARIANTS[variant]
    return [
        Request(
            custom_id=f"item-{row.id}",
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=v["max_tokens"],
                system=SYSTEM,
                messages=[{"role": "user", "content": row.text}],
                thinking=v["thinking"],  # "disabled" is allowed on Haiku 5.5 at effort high or below
                output_config={"effort": v["effort"], "format": {"type": "json_schema", "schema": CLAUDE_SCHEMA}},
            ),
        )
        for row in sample.itertuples()
    ]


def collect(client: anthropic.Anthropic, batch_id: str) -> dict[int, dict[str, object]]:
    """Key results by item id; batch results arrive in any order."""
    out: dict[int, dict[str, object]] = {}
    for result in client.messages.batches.results(batch_id):
        item_id = int(result.custom_id.removeprefix("item-"))
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            out[item_id] = {"raw": text, "pred": parse_label(text), "stop_reason": msg.stop_reason,
                            "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        else:
            out[item_id] = {"raw": f"ERROR: {result.result.type}", "pred": None, "stop_reason": None,
                            "input_tokens": 0, "output_tokens": 0}
    return out


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY missing: put it in .env (see .env.example)")

    variant = sys.argv[1] if len(sys.argv) > 1 else "off"
    if variant not in VARIANTS:
        sys.exit(f"unknown variant {variant!r}; choose one of {sorted(VARIANTS)}")

    client = anthropic.Anthropic()
    sample = pd.read_csv(SAMPLE)
    t0 = time.perf_counter()
    batch = client.messages.batches.create(requests=build_requests(sample, variant))
    print(f"batch {batch.id} created with {len(sample)} requests")

    while batch.processing_status != "ended":
        time.sleep(POLL_SECONDS)
        batch = client.messages.batches.retrieve(batch.id)
        c = batch.request_counts
        print(f"  {batch.processing_status}: processing={c.processing} succeeded={c.succeeded} errored={c.errored}")
    wall = time.perf_counter() - t0

    results = collect(client, batch.id)
    rows = [{"id": i, **results.get(i, {"raw": "ERROR: missing", "pred": None, "stop_reason": None,
                                         "input_tokens": 0, "output_tokens": 0})} for i in sample["id"]]
    out = pd.DataFrame(rows)
    out["eval_count"] = out["output_tokens"]
    out["eval_ns"] = 0  # no per-request decode timing from the API
    out["model"] = f"{MODEL} (Batch API, {VARIANTS[variant]['label']})"
    out["wall_seconds"] = wall  # batch turnaround incl. queueing, not throughput
    cost = (out["input_tokens"].sum() * BATCH_IN_PER_MTOK + out["output_tokens"].sum() * BATCH_OUT_PER_MTOK) / 1e6
    out["api_cost_usd"] = cost / len(out)

    OUT_DIR.mkdir(exist_ok=True)
    suffix = "" if variant == "off" else f"_{variant}"
    out.to_csv(OUT_DIR / f"preds_{MODEL}{suffix}.csv", index=False)
    valid = out["pred"].notna().sum()
    print(f"{MODEL}: {valid}/{len(out)} valid, batch turnaround {wall:.0f}s, "
          f"tokens in={out['input_tokens'].sum()} out={out['output_tokens'].sum()}, est. cost ${cost:.4f}")
    print("stop reasons:", json.dumps(out["stop_reason"].value_counts(dropna=False).to_dict(), default=str))


if __name__ == "__main__":
    main()
