"""
Grade real student essays (ASAP-AES dataset) using ANY OpenAI-compatible LLM API,
blind to the human score, to measure AI-vs-human grading gap.

Works with: Groq, OpenRouter, GitHub Models, Logfare, Cerebras, Mistral, or any
other OpenAI-compatible endpoint. Just set the right env vars for your provider.

SAVES INCREMENTALLY -- safe to Ctrl+C and resume any time with the same --out file.
Already-graded essay_ids are skipped automatically on resume.

Usage:
    # Groq
    export LLM_BASE_URL="https://api.groq.com/openai/v1"
    export LLM_API_KEY="gsk_xxxxxxxxxxxx"
    export LLM_MODEL="llama-3.3-70b-versatile"

    # OpenRouter
    export LLM_BASE_URL="https://openrouter.ai/api/v1"
    export LLM_API_KEY="sk-or-v1-xxxxxxxxxxxx"
    export LLM_MODEL="meta-llama/llama-3.3-70b-instruct:free"

    # GitHub Models (classic PAT: leave ALL scopes unchecked, just generate)
    export LLM_BASE_URL="https://models.github.ai/inference"
    export LLM_API_KEY="ghp_xxxxxxxxxxxx"
    export LLM_MODEL="openai/gpt-4o-mini"

    # Logfare (no signup friction, no rate limits, logs prompts for research -- opt out at logfare.ai/consent)
    export LLM_BASE_URL="https://logfare.ai/v1"
    export LLM_API_KEY="lfu_xxxxxxxxxxxx"
    export LLM_MODEL="kimi-k2.5"

    # Cerebras
    export LLM_BASE_URL="https://api.cerebras.ai/v1"
    export LLM_API_KEY="csk-xxxxxxxxxxxx"
    export LLM_MODEL="llama-3.3-70b"

    # Mistral
    export LLM_BASE_URL="https://api.mistral.ai/v1"
    export LLM_API_KEY="xxxxxxxxxxxx"
    export LLM_MODEL="mistral-large-latest"

    # Google AI Studio (Gemini) -- OpenAI-compatible endpoint
    export LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
    export LLM_API_KEY="AIzaSyxxxxxxxxxxxx"
    export LLM_MODEL="gemini-2.5-flash"

    python3 01_grade_essays_multiprovider.py --input asap_essays.csv --essay-set 1 --n 300 --out ai_grades_set1.csv

If you hit rate limits (Groq free tier is ~30 req/min), increase --sleep, e.g.:
    python3 01_grade_essays_multiprovider.py --input asap_essays.csv --essay-set 1 --n 300 --out ai_grades_set1.csv --sleep 2.5

Requirements:
    pip install openai pandas tenacity --break-system-packages
"""

import argparse
import csv
import os
import sys
import time
import re
import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

RUBRIC_PROMPT = """You are grading a student's persuasive essay written in response to this prompt:
"Write a letter to your local newspaper in which you state your opinion on the effects computers have on people."

Grade the essay on a holistic scale from 0 (lowest) to 10 (highest), based on:
- Organization and structure
- Development and support of ideas (specific examples, reasoning)
- Argument strength and persuasiveness
- Language control (grammar/spelling errors should reduce the score, but do not over-penalize minor typos if the argument is strong)

This is real, unedited student writing — expect some spelling and grammar errors; grade holistically, not just on mechanics.

Respond with ONLY a single integer from 0 to 10. No words, no explanation, no punctuation — just the number.

ESSAY:
{essay}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=3, max=20),
    retry=retry_if_exception_type(Exception),
)
def grade_one(client: OpenAI, model: str, essay_text: str) -> int:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": RUBRIC_PROMPT.format(essay=essay_text)}],
        temperature=0.0,
        max_tokens=5,
    )
    text = resp.choices[0].message.content.strip()
    match = re.search(r"\d+", text)
    if not match:
        raise ValueError(f"Could not parse score from model output: {text!r}")
    score = int(match.group())
    if not (0 <= score <= 10):
        raise ValueError(f"Score out of range: {score}")
    return score


def load_and_clean_previous_run(out_path):
    """
    Read a previous (possibly interrupted) output file, keep only rows that
    actually succeeded (ai_score is not null), and rewrite the file with just
    those -- so failed/N-A rows get purged and will be retried this run,
    instead of being permanently stuck as failures.
    Returns the set of essay_ids that succeeded previously.
    """
    if not os.path.exists(out_path):
        return set()
    try:
        existing = pd.read_csv(out_path)
    except Exception:
        return set()

    successful = existing.dropna(subset=["ai_score"])
    failed_count = len(existing) - len(successful)
    if failed_count > 0:
        print(f"Found {failed_count} previously-failed (N/A) essays in {out_path} -- clearing them so they retry this run.")
        successful.to_csv(out_path, index=False)

    return set(successful["essay_id"].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="asap_essays.csv", help="Path to ASAP essays CSV")
    parser.add_argument("--essay-set", type=int, default=1, help="Which essay_set to grade (1-8)")
    parser.add_argument("--n", type=int, default=300, help="Number of essays to sample and grade")
    parser.add_argument("--out", default="ai_grades.csv", help="Output CSV path (resumes if it already exists)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds to sleep between calls (rate limiting) -- increase if you see repeated failures")
    args = parser.parse_args()

    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")

    missing = [name for name, val in [("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key), ("LLM_MODEL", model)] if not val]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        print("\nExample (Groq):", file=sys.stderr)
        print('  export LLM_BASE_URL="https://api.groq.com/openai/v1"', file=sys.stderr)
        print('  export LLM_API_KEY="gsk_xxxxxxxxxxxx"', file=sys.stderr)
        print('  export LLM_MODEL="llama-3.3-70b-versatile"', file=sys.stderr)
        print("\nSee the top of this script for OpenRouter / GitHub Models / Logfare / Cerebras / Mistral examples.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)

    df = pd.read_csv(args.input)
    subset = df[df.essay_set == args.essay_set].copy()
    if len(subset) == 0:
        print(f"ERROR: No essays found for essay_set={args.essay_set}", file=sys.stderr)
        sys.exit(1)

    n = min(args.n, len(subset))
    sample = subset.sample(n=n, random_state=args.seed).reset_index(drop=True)

    already_done = load_and_clean_previous_run(args.out)
    if already_done:
        print(f"Resuming: found {len(already_done)} successfully-graded essays in {args.out}, skipping those.")

    file_exists = os.path.exists(args.out)
    fieldnames = ["essay_id", "essay_set", "human_score", "ai_score"]

    graded_count = len(already_done)
    failed_count = 0
    consecutive_failures = 0
    current_sleep = args.sleep

    print(f"Grading up to {n} essays from essay_set={args.essay_set} using {model} @ {base_url} ...")
    print(f"Saving incrementally to {args.out} -- safe to Ctrl+C and resume anytime.\n")

    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, row in sample.iterrows():
            essay_id = row["essay_id"]
            if essay_id in already_done:
                continue

            try:
                ai_score = grade_one(client, model, row["essay"])
                consecutive_failures = 0
                current_sleep = max(args.sleep, current_sleep * 0.9)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "RateLimitError" in err_str or "429" in err_str
                print(f"  [{i+1}/{n}] essay_id={essay_id} FAILED after retries: {e}", file=sys.stderr)
                ai_score = None
                failed_count += 1
                consecutive_failures += 1
                if is_rate_limit:
                    current_sleep = min(current_sleep * 2, 30)
                    print(f"    Real rate limit -- increasing delay between requests to {current_sleep:.1f}s", file=sys.stderr)
                elif "NotFoundError" in err_str or "404" in err_str:
                    print(f"    NOT a rate limit -- this looks like an invalid model name. Check LLM_MODEL is a currently valid model ID for this provider.", file=sys.stderr)
                else:
                    print(f"    NOT a rate limit -- check LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are all correct for this provider.", file=sys.stderr)

            writer.writerow({
                "essay_id": essay_id,
                "essay_set": row["essay_set"],
                "human_score": row["final_score"],
                "ai_score": ai_score,
            })
            f.flush()  # write to disk immediately -- survives Ctrl+C / crash

            graded_count += 1
            if graded_count % 10 == 0 or (i + 1) == n:
                print(f"  [{graded_count}/{n}] graded so far (last essay_id={essay_id}, ai_score={ai_score})")

            time.sleep(current_sleep)

    print(f"\nDone. Results saved to {args.out} ({failed_count} failed this run, will retry on next run).")

    out_df = pd.read_csv(args.out).dropna(subset=["ai_score"])
    if len(out_df) > 0:
        gap = out_df["ai_score"] - out_df["human_score"]
        print(f"\nTotal graded so far: {len(out_df)}")
        print(f"Mean human score: {out_df['human_score'].mean():.2f}")
        print(f"Mean AI score:    {out_df['ai_score'].mean():.2f}")
        print(f"Mean gap (AI-human): {gap.mean():.2f}")
        print(f"Correlation: {out_df['human_score'].corr(out_df['ai_score']):.3f}")

    if graded_count < n:
        print(f"\nNote: only {graded_count}/{n} requested essays graded. Run the exact same command again to continue.")


if __name__ == "__main__":
    main()
