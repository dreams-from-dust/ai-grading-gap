"""
Compare grading bias across multiple AI models against real human scores.

Run 01_grade_essays_multiprovider.py once per provider first (into separate
CSVs), then point this script at all of them to compare.

Usage:
    python3 03_compare_models.py \\
        --grades groq=data/ai_grades_groq.csv \\
        --grades openrouter=data/ai_grades_openrouter.csv \\
        --grades gemini=data/ai_grades_gemini.csv \\
        --out data/model_comparison.csv

Requirements:
    pip install pandas scipy --break-system-packages
"""

import argparse
import pandas as pd
from scipy import stats


def parse_grades_arg(items):
    """Parse repeated --grades name=path.csv args into a dict."""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--grades must be name=path.csv, got: {item}")
        name, path = item.split("=", 1)
        out[name] = path
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grades", action="append", required=True,
        help="name=path.csv pair, repeatable, e.g. --grades groq=data/ai_grades_groq.csv"
    )
    parser.add_argument("--out", default="data/model_comparison.csv")
    args = parser.parse_args()

    files = parse_grades_arg(args.grades)
    if len(files) < 2:
        raise SystemExit("Need at least 2 --grades files to compare.")

    frames = {}
    for name, path in files.items():
        df = pd.read_csv(path).dropna(subset=["ai_score"])
        df["gap"] = df["ai_score"] - df["human_score"]
        frames[name] = df
        print(f"\n=== {name} ({path}) ===")
        print(f"  n = {len(df)}")
        print(f"  mean human score: {df['human_score'].mean():.2f}")
        print(f"  mean AI score:    {df['ai_score'].mean():.2f}")
        print(f"  mean gap (AI-human): {df['gap'].mean():+.2f}")
        print(f"  correlation with human: {df['human_score'].corr(df['ai_score']):.3f}")
        print(f"  % over-scored (gap>0): {(df['gap']>0).mean()*100:.1f}%")
        print(f"  % under-scored (gap<0): {(df['gap']<0).mean()*100:.1f}%")

    # Pairwise statistical comparison of gaps (are the biases significantly different?)
    names = list(frames.keys())
    print("\n=== Pairwise t-tests on gap distributions ===")
    print("(tests whether two models' grading bias differs significantly, not just numerically)")
    summary_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            merged = frames[a][["row_id", "gap"]].merge(
                frames[b][["row_id", "gap"]], on="row_id", suffixes=(f"_{a}", f"_{b}")
            )
            if len(merged) < 2:
                print(f"  {a} vs {b}: not enough overlapping essay_ids to compare directly")
                continue
            t_stat, p_val = stats.ttest_rel(merged[f"gap_{a}"], merged[f"gap_{b}"])
            sig = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "not significant"
            print(f"  {a} vs {b}: n_overlap={len(merged)}, t={t_stat:.3f}, p={p_val:.4f} -> {sig}")
            summary_rows.append({
                "model_a": a, "model_b": b, "n_overlap": len(merged),
                "t_stat": t_stat, "p_value": p_val, "significant": p_val < 0.05
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(args.out, index=False)
        print(f"\nComparison summary saved to {args.out}")
    else:
        print("\nNo overlapping essay_ids found between files -- run all providers on the SAME "
              "--seed and --essay-set so they grade the same essays for a fair comparison.")


if __name__ == "__main__":
    main()
