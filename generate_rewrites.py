#!/usr/bin/env python3
"""Generate five-condition rewrites for the fixed E2 seeds.

This is the L2-style skeleton: one independent call per (seed, condition,
candidate). The L3 repair loop is intentionally not implemented; feed the
validator failures back with the template in rewrite_prompts.md section 5.2.

Providers:
  mock    deterministic local stub, for pipeline smoke tests only
  openai  any OpenAI-compatible chat endpoint (OpenAI, vLLM, Ollama, LM Studio)

Examples:
  python generate_rewrites.py --provider mock --limit-seeds 2 --candidates 1 --validate
  python generate_rewrites.py --dry-run --limit-seeds 1
  OPENAI_API_KEY=... python generate_rewrites.py --provider openai \
      --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_CONDITIONS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]


# ---------------------------------------------------------------------------
# prompt loading
# ---------------------------------------------------------------------------
def first_text_block(section: str) -> str:
    match = re.search(r"```text\n(.*?)\n```", section, re.S)
    if not match:
        raise ValueError("no ```text block found in section")
    return match.group(1).strip()


def load_prompts(path: Path) -> tuple[str, dict[str, str]]:
    markdown = path.read_text(encoding="utf-8")

    system_heading = re.search(r"^## 2\. 共享 System Prompt\s*$", markdown, re.M)
    if not system_heading:
        raise ValueError("cannot find '## 2. 共享 System Prompt'")
    rest = markdown[system_heading.end():]
    next_h2 = re.search(r"^## ", rest, re.M)
    system = first_text_block(rest[:next_h2.start()] if next_h2 else rest)

    conditions: dict[str, str] = {}
    heading_re = re.compile(r"^### 3\.\d+ `([a-z_]+)`[^\n]*$", re.M)
    for match in heading_re.finditer(markdown):
        name = match.group(1)
        if name not in ALLOWED_CONDITIONS:
            continue
        section = markdown[match.end():]
        next_heading = re.search(r"^#{2,3} ", section, re.M)
        conditions[name] = first_text_block(
            section[:next_heading.start()] if next_heading else section
        )

    missing = [c for c in ALLOWED_CONDITIONS if c not in conditions]
    if missing:
        raise ValueError(f"missing condition prompts: {missing}")
    return system, conditions


def build_input(seed: dict, condition: str) -> dict:
    return {
        "condition": condition,
        "language": seed.get("language", "en"),
        "subject": seed.get("subject", ""),
        "body": seed.get("body", ""),
        "protected_spans": seed.get("protected_spans", []),
        "notes": seed.get("notes", ""),
    }


def build_user_message(condition_prompt: str, payload: dict) -> str:
    return (
        f"{condition_prompt}\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        'Return only:\n{"subject": "...", "body": "..."}'
    )


# ---------------------------------------------------------------------------
# output parsing
# ---------------------------------------------------------------------------
def _loads_tolerant(text: str):
    for kwargs in ({}, {"strict": False}):
        try:
            return json.loads(text, **kwargs)
        except json.JSONDecodeError:
            continue
    if text.startswith("{") and not text.rstrip().endswith("}"):
        # Small local models often stop one brace short after a long body.
        for suffix in ("}", '"\n}'):
            try:
                return json.loads(text.rstrip() + suffix, strict=False)
            except json.JSONDecodeError:
                continue
    if text.startswith("{") and text.rstrip().endswith("}") and not text.rstrip().endswith('"}'):
        # ...or close the brace but forget the body string's closing quote.
        try:
            return json.loads(text.rstrip()[:-1].rstrip() + '"\n}', strict=False)
        except json.JSONDecodeError:
            pass
    return None


def parse_output(raw: str) -> tuple[str, str] | None:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    obj = _loads_tolerant(text)
    if obj is None:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            obj = _loads_tolerant(text[start:end + 1])
    if obj is None:
        # Last resort: raw newlines or unescaped quotes inside body. Subject is
        # non-greedy; body runs to the final quote before the closing brace.
        match = re.search(
            r'"subject"\s*:\s*"(.*?)"\s*,\s*"body"\s*:\s*"(.*)"\s*\}\s*$',
            text, re.S,
        )
        if match:
            return match.group(1), match.group(2)
        return None
    if not isinstance(obj, dict):
        return None
    subject, body = obj.get("subject"), obj.get("body")
    if not isinstance(subject, str) or not isinstance(body, str):
        return None
    return subject, body


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
class MockProvider:
    """Deterministic stub: echoes the input with a visible mock marker."""

    name = "mock"

    def __init__(self, model: str = "mock-model"):
        self.model = model

    def generate(self, system: str, user: str, *, temperature: float, top_p: float,
                 random_seed: int | None, max_tokens: int) -> str:
        match = re.search(r"INPUT:\n(\{.*?\})\n\nReturn only:", user, re.S)
        if not match:
            raise ValueError("mock provider could not find INPUT JSON")
        payload = json.loads(match.group(1))
        body = f"[mock {payload['condition']}] {payload['body']}"
        return json.dumps(
            {"subject": f"[mock {payload['condition']}] {payload['subject']}", "body": body},
            ensure_ascii=False,
        )


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, api_key: str, base_url: str | None,
                 disable_thinking: bool = False, reasoning_effort: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "openai package is required for --provider openai; run: pip install openai"
            ) from exc
        if not model:
            raise SystemExit("--model is required for --provider openai")
        self.model = model
        self.disable_thinking = disable_thinking
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def generate(self, system: str, user: str, *, temperature: float, top_p: float,
                 random_seed: int | None, max_tokens: int) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if random_seed is not None:
            kwargs["seed"] = random_seed
        if self.disable_thinking:
            # DeepSeek reasoning models spend max_tokens on reasoning_content
            # and can return an empty message; disable thinking for rewrites.
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def make_provider(args: argparse.Namespace):
    if args.provider == "mock":
        return MockProvider(args.model or "mock-model")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("set OPENAI_API_KEY or pass --api-key")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    return OpenAIProvider(args.model, api_key, base_url,
                          disable_thinking=args.disable_thinking,
                          reasoning_effort=args.reasoning_effort)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def request_seed(seed_base: int, seed_index: int, condition_index: int, candidate: int) -> int:
    return seed_base + seed_index * 100 + condition_index * 10 + candidate


def load_done(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error") or row.get("parsed_ok") is False:
                continue  # retry failed/unparsed requests on --resume
            done.add((str(row.get("seed_id", "")), str(row.get("condition", "")),
                      int(row.get("candidate", 0))))
    return done


def auto_max_tokens(body: str) -> int:
    # Qwen pretty-prints JSON with long URLs and can spend ~9 tokens per
    # original word. Reserve 5x + 800; seed bodies are <= 300 words, so this
    # stays inside the local model's 4096-token context.
    return max(1000, int(len(body.split()) * 5.0) + 800)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate five-condition rewrites.")
    parser.add_argument("--seeds", default="samples/e2_seed_inputs.jsonl")
    parser.add_argument("--prompts", default="prompts/rewrite_prompts.md")
    parser.add_argument("--out", default="generation_log.jsonl")
    parser.add_argument("--provider", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--quantization", default="")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="send extra_body thinking=disabled (DeepSeek reasoning models)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                        default=None, help="provider reasoning budget hint")
    parser.add_argument("--conditions", nargs="+", default=ALLOWED_CONDITIONS)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--limit-seeds", type=int, default=0, help="0 = all seeds")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 = auto from body length")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conditions: list[str] = []
    for item in args.conditions:
        for name in item.split(","):
            name = name.strip()
            if name and name not in conditions:
                conditions.append(name)
    args.conditions = conditions
    if not args.conditions:
        raise SystemExit("no conditions selected")
    unknown = [c for c in args.conditions if c not in ALLOWED_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}; allowed: {ALLOWED_CONDITIONS}")
    if args.candidates < 1:
        raise SystemExit("--candidates must be >= 1")

    system, condition_prompts = load_prompts(Path(args.prompts))

    seeds: list[dict] = []
    with Path(args.seeds).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seeds.append(json.loads(line))
    if args.limit_seeds > 0:
        seeds = seeds[:args.limit_seeds]
    if not seeds:
        raise SystemExit("no seeds loaded")

    if args.dry_run:
        seed = seeds[0]
        for condition in args.conditions:
            payload = build_input(seed, condition)
            user = build_user_message(condition_prompts[condition], payload)
            digest = hashlib.sha256(f"{system}\n\n{user}".encode()).hexdigest()
            print(json.dumps({
                "seed_id": seed.get("seed_id"),
                "condition": condition,
                "prompt_sha256": digest,
                "system": system,
                "user": user,
            }, ensure_ascii=False, indent=2))
            print("\n" + "=" * 80 + "\n")
        return

    provider = make_provider(args)
    out_path = Path(args.out)
    done = load_done(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    validator = None
    if args.validate:
        from validate_rewrites import validate_pair

        validator = validate_pair

    total = len(seeds) * len(args.conditions) * args.candidates
    written = skipped = errors = 0

    with out_path.open(mode, encoding="utf-8") as out:
        for seed_index, seed in enumerate(seeds):
            seed_id = str(seed.get("seed_id", f"seed-{seed_index}"))
            body = str(seed.get("body", ""))
            max_tokens = args.max_tokens or auto_max_tokens(body)

            for condition_index, condition in enumerate(args.conditions):
                payload = build_input(seed, condition)
                user = build_user_message(condition_prompts[condition], payload)
                prompt_sha = hashlib.sha256(f"{system}\n\n{user}".encode()).hexdigest()

                for candidate in range(1, args.candidates + 1):
                    key = (seed_id, condition, candidate)
                    if key in done:
                        skipped += 1
                        continue

                    random_seed = request_seed(
                        args.seed_base, seed_index, condition_index, candidate
                    )
                    record = {
                        "pair_id": seed_id,
                        "seed_id": seed_id,
                        "condition": condition,
                        "provider": provider.name,
                        "model": provider.model,
                        "quantization": args.quantization,
                        "reasoning": "disabled" if args.disable_thinking else "default",
                        "reasoning_effort": args.reasoning_effort or "",
                        "prompt_sha256": prompt_sha,
                        "seed": random_seed,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": max_tokens,
                        "candidate": candidate,
                        "attempt": 1,
                        "subject_original": seed.get("subject", ""),
                        "original": body,
                        "raw_output": "",
                        "parsed_ok": False,
                        "parsed_subject": "",
                        "parsed_body": "",
                        "paraphrase": "",
                        "subject_paraphrase": "",
                        "validator_pass": None,
                        "validator_failures": "",
                        "validator_warnings": "",
                        "length_ratio": None,
                        "error": "",
                        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }

                    try:
                        raw = provider.generate(
                            system, user,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            random_seed=random_seed,
                            max_tokens=max_tokens,
                        )
                        record["raw_output"] = raw
                        parsed = parse_output(raw)
                        if parsed is None:
                            record["error"] = "parsed_ok=false"
                        else:
                            subject_paraphrase, body_paraphrase = parsed
                            record.update({
                                "parsed_ok": True,
                                "parsed_subject": subject_paraphrase,
                                "parsed_body": body_paraphrase,
                                "paraphrase": body_paraphrase,
                                "subject_paraphrase": subject_paraphrase,
                            })
                            if validator is not None:
                                try:
                                    result = validator(
                                        original=body,
                                        paraphrase=body_paraphrase,
                                        subject_original=str(seed.get("subject", "")),
                                        subject_paraphrase=subject_paraphrase,
                                        condition=condition,
                                    )
                                    record.update({
                                        "validator_pass": bool(result.get("pass")),
                                        "validator_failures": result.get("failure_codes", ""),
                                        "validator_warnings": result.get("warnings", ""),
                                        "length_ratio": result.get("length_ratio"),
                                    })
                                except Exception as exc:  # keep the raw generation
                                    record["error"] = f"validator_error: {exc}"
                    except Exception as exc:  # log and continue the batch
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        errors += 1

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    written += 1
                    print(f"[{written + skipped}/{total}] {seed_id} {condition} cand={candidate}"
                          f" parsed={record['parsed_ok']} pass={record['validator_pass']}"
                          f" error={record['error'][:80]}")

    print(f"\ndone: written={written} skipped={skipped} errors={errors} out={out_path}")


if __name__ == "__main__":
    main()
