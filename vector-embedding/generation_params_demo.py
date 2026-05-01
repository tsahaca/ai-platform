"""
Generation Parameters Demo
---------------------------
Demonstrates how sampling parameters — temperature, top-p (nucleus sampling),
and top-k — affect text generation using GPT-2 via HuggingFace Transformers.

Parameters explained
---------------------
temperature : float  (0 < t ≤ ∞)
    Scales the logits before softmax.
    • Low  (e.g. 0.2) → sharper distribution, more deterministic / repetitive output.
    • High (e.g. 1.5) → flatter distribution, more diverse / creative output.

top_k : int  (k ≥ 1)
    At each step, only the k highest-probability tokens are considered.
    • k=1  → greedy decoding (always pick the most likely token).
    • k=50 → keeps a broad but bounded candidate set.

top_p : float  (0 < p ≤ 1)
    Nucleus sampling — keeps the smallest set of tokens whose cumulative
    probability ≥ p, then renormalises.
    • p=1.0  → no filtering (use all tokens).
    • p=0.9  → trim the long tail, keep 90 % of probability mass.

top_k and top_p can be combined; both filters are applied together.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ── config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "gpt2"
PROMPT      = "Once upon a time in a kingdom far away"
MAX_NEW_TOKENS = 60
SEED        = 42          # set None to see different results each run

# Scenarios that highlight the effect of each parameter
SCENARIOS = [
    {
        "label":       "Greedy (baseline)",
        "description": "No sampling — always pick the most probable token.",
        "do_sample":   False,
    },
    {
        "label":       "Low temperature  (t=0.3)",
        "description": "Very conservative / focused output.",
        "do_sample":   True,
        "temperature": 0.3,
        "top_k":       0,
        "top_p":       1.0,
    },
    {
        "label":       "Medium temperature  (t=1.0)",
        "description": "Default sampling — balanced creativity.",
        "do_sample":   True,
        "temperature": 1.0,
        "top_k":       0,
        "top_p":       1.0,
    },
    {
        "label":       "High temperature  (t=1.5)",
        "description": "More random / creative output.",
        "do_sample":   True,
        "temperature": 1.5,
        "top_k":       0,
        "top_p":       1.0,
    },
    {
        "label":       "top-k=10",
        "description": "Only the 10 most likely tokens are candidates at each step.",
        "do_sample":   True,
        "temperature": 1.0,
        "top_k":       10,
        "top_p":       1.0,
    },
    {
        "label":       "top-k=50",
        "description": "Broader top-k candidate set.",
        "do_sample":   True,
        "temperature": 1.0,
        "top_k":       50,
        "top_p":       1.0,
    },
    {
        "label":       "top-p=0.7  (nucleus)",
        "description": "Keep tokens summing to 70 % probability mass.",
        "do_sample":   True,
        "temperature": 1.0,
        "top_k":       0,
        "top_p":       0.7,
    },
    {
        "label":       "top-p=0.95  (nucleus)",
        "description": "Keep tokens summing to 95 % probability mass.",
        "do_sample":   True,
        "temperature": 1.0,
        "top_k":       0,
        "top_p":       0.95,
    },
    {
        "label":       "Combined  (t=0.8, top-k=40, top-p=0.9)",
        "description": "Typical production setting — moderate and coherent.",
        "do_sample":   True,
        "temperature": 0.8,
        "top_k":       40,
        "top_p":       0.9,
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────

def generate(model, tokenizer, scenario: dict, seed: int | None) -> str:
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Build kwargs — only pass sampling params when do_sample=True
    gen_kwargs: dict = dict(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=scenario.get("do_sample", False),
        pad_token_id=tokenizer.eos_token_id,
    )
    if scenario.get("do_sample"):
        gen_kwargs["temperature"] = scenario.get("temperature", 1.0)
        gen_kwargs["top_k"]       = scenario.get("top_k", 50)
        gen_kwargs["top_p"]       = scenario.get("top_p", 1.0)

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    # Decode only the newly generated tokens (strip the prompt)
    new_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Generation Parameters Demo  —  GPT-2")
    print("=" * 70)
    print(f"\nPrompt : \"{PROMPT}\"")
    print(f"Seed   : {SEED}\n")

    print("Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.\n")

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"{'─' * 70}")
        print(f"[{i}/{len(SCENARIOS)}]  {scenario['label']}")
        print(f"         {scenario['description']}")

        # Show the active parameters
        params = {k: v for k, v in scenario.items()
                  if k not in ("label", "description")}
        print(f"         params : {params}")

        text = generate(model, tokenizer, scenario, SEED)
        print(f"\n  {PROMPT}{text}\n")

    print("=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
