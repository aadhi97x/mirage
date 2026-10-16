"""
Plain-language wording for investigations, written by a LOCAL open-source LLM served by Ollama (https://ollama.com).
Nothing leaves the machine and no API key is needed.

Rules for the model: it receives only the already-calculated facts of ONE investigation (never the raw CSVs), must restate those
numbers and may not invent new ones, must not claim causation, and must not call anything "profit".
All figures come from deterministic pandas / scikit-learn code (traps.py, ml.py); the LLM only phrases them.

If Ollama is not running (for example on a hosted deployment) the app falls back to a plain template and says so in the UI.

Config (optional environment variables):
  OLLAMA_HOST       default http://localhost:11434
  MONEY_LEAK_LLM    model name to use, default: first installed of PREFERRED
"""
import json
import os
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
PREFERRED = ["qwen2.5:7b", "qwen2.5:7b-instruct", "llama3.1:8b", "llama3.2:3b", "qwen2.5:3b", "phi3:mini"]


def installed_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=1.5) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


def pick_model() -> str | None:
    """Env override, else first preferred model that is installed, else any installed model."""
    names = installed_models()
    forced = os.environ.get("MONEY_LEAK_LLM")
    if forced:
        return forced if (not names or forced in names) else None
    return next((m for m in PREFERRED if m in names), names[0] if names else None)


def status() -> tuple[bool, str]:
    m = pick_model()
    return (True, m) if m else (False, "not running")


def _generate(prompt: str, model: str) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.2, "num_predict": 400}}).encode()
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:  # first call loads the model into memory
        return json.loads(r.read())["response"].strip()


def card_facts(card: dict) -> str:
    ev = "\n".join(f"- {e}" for e in card["evidence"])
    unk = "\n".join(f"- {u}" for u in card.get("unknown", card["uncertainty"]))
    return (f"Investigation: {card['finding_title']} (status: {card['status']})\n"
            f"What looks good on the surface: {card['headline']}\n"
            f"What the records show: {card['concern']}\n"
            f"Evidence:\n{ev}\n"
            f"Financial figure ({card['impact_label']}): {card['impact_text']}\n"
            f"What the data does not tell us:\n{unk}\n"
            f"Next step: {card['next_step']}")


def _run(prompt: str, template: str) -> tuple[str, str]:
    model = pick_model()
    if not model:
        return template, "template (local LLM not running; start Ollama to enable)"
    try:
        text = _generate(prompt, model)
        if text:
            return text, f"local LLM via Ollama ({model})"
    except Exception as e:
        return template, f"template (local LLM call failed: {type(e).__name__})"
    return template, "template (local LLM returned nothing)"


def explain(card: dict) -> tuple[str, str]:
    """(text, source label) for one investigation."""
    prompt = ("You are a careful business analyst writing for a store manager with no analytics background. "
              "Using ONLY the facts below, write at most 120 words in plain language: (1) the contradiction, (2) the strongest evidence, "
              "(3) what is still uncertain, (4) the next thing to check. Rules: do not add any number that is not in the facts; "
              "do not claim causation; do not use the word 'profit' (use 'margin proxy').\n\n" + card_facts(card))
    template = f"{card['finding_title']}. {card['evidence'][0]} Next: {card['next_step']}"
    return _run(prompt, template)


def brief(cards: list) -> tuple[str, str]:
    """One short brief across all investigations."""
    flagged = [c for c in cards if c["status"] == "Flagged"]
    tested = [c for c in cards if c["status"] != "Flagged"]
    facts = "\n\n".join(card_facts(c) for c in flagged)
    nots = "; ".join(f"{c['name']}: {c['evidence'][0]}" for c in tested)
    prompt = ("You are a careful business analyst writing a short brief for a store manager. Using ONLY the facts below, write at most 160 words: "
              "start with the most important contradiction, then the second, then one sentence on what was tested and NOT supported, then the first action. "
              "Do not add numbers that are not in the facts, do not claim causation, say 'margin proxy' not 'profit'.\n\n"
              f"FLAGGED:\n{facts}\n\nTESTED BUT NOT SUPPORTED:\n{nots}")
    template = " ".join(f"{c['finding_title']}." for c in flagged) + " Next: " + flagged[0]["next_step"]
    return _run(prompt, template)
