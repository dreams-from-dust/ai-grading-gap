"""
Grade real student SHORT-ANSWER responses (ASAG dataset, CS/technical questions)
using any OpenAI-compatible LLM API, blind to human score, to measure AI-vs-human
grading gap in this second domain (complements essay grading).

Unlike essays, each short-answer response has a specific question + reference
answer, so the grader is given both -- this mirrors how a real short-answer
grading rubric works (compare student answer against a reference answer).

SAVES INCREMENTALLY -- safe to Ctrl+C and resume any time with the same --out file.

Usage:
    export LLM_BASE_URL="https://api.groq.com/openai/v1"
    export LLM_API_KEY="gsk_xxxxxxxxxxxx"
    export LLM_MODEL="llama-3.3-70b-versatile"

    python3 04_grade_short_answers.py --input short_answers.csv --n 300 --out ai_grades_shortanswer_groq.csv

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

RUBRIC_PROMPT = """You are grading a student's short-answer response to a technical/academic question.

QUESTION:
{question}

REFERENCE ANSWER (what a full-credit answer should cover):
{ref_answer}

Grade the student's answer on a scale from 0 (no credit -- missing or wrong) to 2 (full credit --
correct and complete), where 1 means partial credit (some correct content but incomplete or with errors).

This is real, unedited student writing. Compare the STUDENT ANSWER below against the reference answer
above and judge how much of the required content it captures, not exact wording.

Respond with ONLY a single integer: 0, 1, or 2. No words, no explanation, no punctuation.

STUDENT ANSWER:
{answer}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=3, max=20),
    retry=retry_if_exception_type(Exception),
)
def grade_one(client: OpenAI, model: str, question: str, ref_answer: str, student_answer: str) -> int:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": RUBRIC_PROMPT.format(
            question=question, ref_answer=ref_answer, answer=student_answer
        )}],
        temperature=0.0,
        max_tokens=5,
    )
    text = resp.choices[0].message.content.strip()
    match = re.search(r"\d+", text)
    if not match:
        raise ValueError(f"Could not parse score from model output: {text!r}")
    score = int(match.group())
    if not (0 <= score <= 2):
        raise ValueError(f"Score out of range: {score}")
    return score


def load_and_clean_previous_run(out_path):
    """
    Purge previously-failed (N/A) rows from a prior run so they retry this run,
    instead of being permanently stuck. Returns row_ids that succeeded before.
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
        print(f"Found {failed_count} previously-failed (N/A) responses in {out_path} -- clearing them so they retry this run.")
        successful.to_csv(out_path, index=False)

    return set(successful["row_id"].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="short_answers.csv")
    parser.add_argument("--n", type=int, default=300, help="Number of responses to sample and grade")
    parser.add_argument("--out", default="ai_grades_shortanswer.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds between calls (rate limiting) -- increase if you see repeated failures")
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
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)

    df = pd.read_csv(args.input)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index  # stable id for resuming

    n = min(args.n, len(df))
    sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    already_done = load_and_clean_previous_run(args.out)
    if already_done:
        print(f"Resuming: found {len(already_done)} successfully-graded responses in {args.out}, skipping those.")

    file_exists = os.path.exists(args.out)
    fieldnames = ["row_id", "question_id", "human_score", "ai_score"]

    graded_count = len(already_done)
    failed_count = 0
    current_sleep = args.sleep

    print(f"Grading up to {n} short-answer responses using {model} @ {base_url} ...")
    print(f"Saving incrementally to {args.out} -- safe to Ctrl+C and resume anytime.\n")

    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, row in sample.iterrows():
            row_id = row["row_id"]
            if row_id in already_done:
                continue

            try:
                ai_score = grade_one(client, model, row["question"], row["ref_answer"], row["student_answer"])
                current_sleep = max(args.sleep, current_sleep * 0.9)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "RateLimitError" in err_str or "429" in err_str
                print(f"  [{i+1}/{n}] row_id={row_id} FAILED after retries: {e}", file=sys.stderr)
                ai_score = None
                failed_count += 1
                if is_rate_limit:
                    current_sleep = min(current_sleep * 2, 30)
                    print(f"    Real rate limit -- increasing delay to {current_sleep:.1f}s", file=sys.stderr)
                elif "NotFoundError" in err_str or "404" in err_str:
                    print(f"    NOT a rate limit -- this looks like an invalid model name for this provider.", file=sys.stderr)
                else:
                    print(f"    NOT a rate limit -- check LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.", file=sys.stderr)

            writer.writerow({
                "row_id": row_id,
                "question_id": row["question_id"],
                "human_score": row["human_score"],
                "ai_score": ai_score,
            })
            f.flush()

            graded_count += 1
            if graded_count % 10 == 0 or (i + 1) == n:
                print(f"  [{graded_count}/{n}] graded so far (last row_id={row_id}, ai_score={ai_score})")

            time.sleep(current_sleep)

    print(f"\nDone. Results saved to {args.out} ({failed_count} failed this run).")

    out_df = pd.read_csv(args.out).dropna(subset=["ai_score"])
    if len(out_df) > 0:
        gap = out_df["ai_score"] - out_df["human_score"]
        print(f"\nTotal graded so far: {len(out_df)}")
        print(f"Mean human score: {out_df['human_score'].mean():.2f}")
        print(f"Mean AI score:    {out_df['ai_score'].mean():.2f}")
        print(f"Mean gap (AI-human): {gap.mean():.2f}")
        print(f"Correlation: {out_df['human_score'].corr(out_df['ai_score']):.3f}")

    if graded_count < n:
        print(f"\nNote: only {graded_count}/{n} requested responses graded. Run the exact same command again to continue.")


if __name__ == "__main__":
    main()
