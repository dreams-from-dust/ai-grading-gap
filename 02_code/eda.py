"""
Exploratory Data Analysis (EDA) for both domains: essays (ASAP-AES) and
short answers (ASAG). Run this BEFORE grading to understand the data you're
working with -- distributions, missing values, feature correlations with
human scores, and any red flags that would affect the grading pipeline.

Usage:
    python3 05_eda.py --essays asap_essays.csv --short-answers short_answers.csv --outdir eda_output

Requirements:
    pip install pandas matplotlib seaborn --break-system-packages
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def eda_essays(path, outdir):
    print("=" * 60)
    print("ESSAY DATASET (ASAP-AES) -- EXPLORATORY ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"\nShape: {df.shape[0]} essays, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    print(f"\nEssay sets (prompts): {sorted(df.essay_set.unique())}")
    print("\nEssays per set:")
    print(df.essay_set.value_counts().sort_index())

    print("\nScore range per essay set:")
    print(df.groupby("essay_set")["final_score"].agg(["min", "max", "mean", "count"]).round(2))

    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None -- clean dataset.")

    print("\nWord count stats (essay_set 1, the domain we're grading):")
    s1 = df[df.essay_set == 1]
    print(s1["word_count"].describe().round(1))

    print("\nSpelling error stats (essay_set 1):")
    print(s1["spell_err_count"].describe().round(2))

    print("\nCorrelation of features with human score (essay_set 1):")
    features = ["char_count", "word_count", "sent_count", "avg_word_len",
                "spell_err_count", "noun_count", "adj_count", "verb_count", "adv_count"]
    corrs = s1[features + ["final_score"]].corr()["final_score"].drop("final_score").sort_values(key=abs, ascending=False)
    print(corrs.round(3))

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(s1["final_score"], bins=range(0, 12), edgecolor="black", color="#4C72B0")
    axes[0, 0].set_title("Essay Set 1: Human Score Distribution")
    axes[0, 0].set_xlabel("Score (0-10)")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(s1["word_count"], bins=30, edgecolor="black", color="#DD8452")
    axes[0, 1].set_title("Essay Set 1: Word Count Distribution")
    axes[0, 1].set_xlabel("Word count")

    axes[1, 0].scatter(s1["word_count"], s1["final_score"], alpha=0.3, color="#55A868")
    axes[1, 0].set_title(f"Word Count vs Human Score (r={s1['word_count'].corr(s1['final_score']):.2f})")
    axes[1, 0].set_xlabel("Word count")
    axes[1, 0].set_ylabel("Human score")

    axes[1, 1].scatter(s1["spell_err_count"], s1["final_score"], alpha=0.3, color="#C44E52")
    axes[1, 1].set_title(f"Spelling Errors vs Human Score (r={s1['spell_err_count'].corr(s1['final_score']):.2f})")
    axes[1, 1].set_xlabel("Spelling error count")
    axes[1, 1].set_ylabel("Human score")

    plt.tight_layout()
    out_path = os.path.join(outdir, "eda_essays.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved plots to {out_path}")


def eda_short_answers(path, outdir):
    print("\n" + "=" * 60)
    print("SHORT-ANSWER DATASET (ASAG) -- EXPLORATORY ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"\nShape: {df.shape[0]} responses, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    print(f"\nUnique questions: {df.question_id.nunique()}")
    print("\nResponses per question:")
    print(df.question_id.value_counts().sort_index())

    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None -- clean dataset.")
    blank_count = df["student_answer"].isna().sum()
    print(f"\n-> {blank_count} responses ({blank_count/len(df)*100:.1f}%) have a BLANK student answer.")
    print("   These are handled by the grading script as automatic 0-credit, no API call wasted.")

    print("\nHuman score distribution (0-2 scale):")
    print(df.human_score.value_counts().sort_index())
    print(f"Mean score: {df.human_score.mean():.3f}")

    df_valid = df.dropna(subset=["student_answer"]).copy()
    df_valid["answer_word_count"] = df_valid["student_answer"].str.split().str.len()
    df_valid["ref_word_count"] = df_valid["ref_answer"].str.split().str.len()
    df_valid["length_ratio"] = df_valid["answer_word_count"] / df_valid["ref_word_count"]

    print("\nAnswer word count stats (excluding blanks):")
    print(df_valid["answer_word_count"].describe().round(1))

    print("\nCorrelation of answer length with human score:")
    print(f"  word count vs score: r = {df_valid['answer_word_count'].corr(df_valid['human_score']):.3f}")
    print(f"  length ratio (student/reference) vs score: r = {df_valid['length_ratio'].corr(df_valid['human_score']):.3f}")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(df["human_score"], bins=[-0.5, 0.5, 1.5, 2.5], edgecolor="black", color="#4C72B0", rwidth=0.8)
    axes[0, 0].set_title("Short Answers: Human Score Distribution (0-2)")
    axes[0, 0].set_xticks([0, 1, 2])
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(df_valid["answer_word_count"], bins=30, edgecolor="black", color="#DD8452")
    axes[0, 1].set_title("Short Answers: Word Count Distribution")
    axes[0, 1].set_xlabel("Word count")

    axes[1, 0].scatter(df_valid["answer_word_count"], df_valid["human_score"], alpha=0.3, color="#55A868")
    axes[1, 0].set_title("Answer Length vs Human Score")
    axes[1, 0].set_xlabel("Word count")
    axes[1, 0].set_ylabel("Human score")

    q_means = df.groupby("question_id")["human_score"].mean().sort_values()
    axes[1, 1].barh(q_means.index.astype(str), q_means.values, color="#8172B2")
    axes[1, 1].set_title("Mean Human Score by Question")
    axes[1, 1].set_xlabel("Mean score (0-2)")
    axes[1, 1].set_ylabel("Question ID")

    plt.tight_layout()
    out_path = os.path.join(outdir, "eda_short_answers.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved plots to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--essays", default="asap_essays.csv")
    parser.add_argument("--short-answers", default="short_answers.csv")
    parser.add_argument("--outdir", default="eda_output")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    eda_essays(args.essays, args.outdir)
    eda_short_answers(args.short_answers, args.outdir)

    print("\n" + "=" * 60)
    print(f"EDA complete. Plots saved in ./{args.outdir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
