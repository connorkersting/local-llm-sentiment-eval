"""One synchronous request with the batch's exact params: checks the key, model id and parameters before a full run."""
import pandas as pd
import anthropic
from dotenv import load_dotenv

from run_eval import SAMPLE, parse_label
from run_eval_claude import ROOT, build_requests

load_dotenv(ROOT / ".env")
req = build_requests(pd.read_csv(SAMPLE).head(1))[0]
msg = anthropic.Anthropic().messages.create(**req["params"])
text = next((b.text for b in msg.content if b.type == "text"), "")
print(f"stop={msg.stop_reason} raw={text!r} parsed={parse_label(text)} "
      f"in={msg.usage.input_tokens} out={msg.usage.output_tokens}")
