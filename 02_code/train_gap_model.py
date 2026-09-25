"""
Train a model that predicts the AI-vs-human grading GAP from response features.

Works for EITHER domain:
  - Essays: uses linguistic features already in asap_essays.csv (word count, spelling
    errors, POS counts).
  - Short answers: features are computed on the fly from the response text itself
    (length, word count) since the ASAG dataset doesn't ship precomputed POS features.

Usage (essays):
    python3 02_train_gap_model.py --domain essay --essays asap_essays.csv --grades ai_grades_groq.csv --out gap_model_essay.pkl

Usage (short answers):
    python3 02_train_gap_model.py --domain shortanswer --essays short_answers.csv --grades ai_grades_shortanswer_groq.csv --out gap_model_sa.pkl

Requirements:
    pip install scikit-learn pandas joblib --break-system-packages
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

ESSAY_FEATURES = [
    "char_count", "word_count", "sent_count", "avg_word_len",
    "spell_err_count", "noun_count", "adj_count", "verb_count", "adv_count",
]


def build_short_answer_features(df):
    """Compute simple text features for short answers (no precomputed POS available)."""
    df = df.copy()
    df["char_count"] = df["student_answer"].astype(str).str.len()
    df["word_count"] = df["student_answer"].astype(str).str.split().str.len()
    df["ref_word_count"] = df["ref_answer"].astype(str).str.split().str.len()
    df["length_ratio"] = df["word_count"] / df["ref_word_count"].replace(0, 1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["essay", "shortanswer"], required=True)
    parser.add_argument("--essays", required=True, help="Path to the source data CSV (asap_essays.csv or short_answers.csv)")
    parser.add_argument("--grades", required=True, help="Path to the AI grades CSV from step 1")
    parser.add_argument("--out", default="gap_model.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = pd.read_csv(args.essays)
    grades = pd.read_csv(args.grades).dropna(subset=["ai_score"])

    if args.domain == "essay":
        df = grades.merge(source[["essay_id"] + ESSAY_FEATURES], on="essay_id", how="left")
        features = ESSAY_FEATURES
    else:
        source = source.reset_index(drop=True)
        source["row_id"] = source.index
        source = build_short_answer_features(source)
        features = ["char_count", "word_count", "ref_word_count", "length_ratio"]
        df = grades.merge(source[["row_id"] + features], on="row_id", how="left")

    df["gap"] = df["ai_score"] - df["human_score"]

    print(f"Domain: {args.domain}")
    print(f"Loaded {len(df)} graded responses.")
    print(f"\nGap distribution (AI score - human score):")
    print(df["gap"].describe())
    print(f"\n% over-scored (gap > 0): {(df['gap'] > 0).mean()*100:.1f}%")
    print(f"% under-scored (gap < 0): {(df['gap'] < 0).mean()*100:.1f}%")
    print(f"% exact match (gap == 0): {(df['gap'] == 0).mean()*100:.1f}%")

    X = df[features]
    y = df["gap"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    model = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=args.seed
    )

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    print(f"\n5-fold CV MAE on training set: {-cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\nTest set performance:")
    print(f"  MAE: {mae:.3f} points")
    print(f"  R^2: {r2:.3f}")

    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    print(f"  Baseline MAE (always predict mean gap): {baseline_mae:.3f}")
    print(f"  Improvement over baseline: {(1 - mae/baseline_mae)*100:.1f}%")

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\nFeature importances (what predicts the AI-human gap):")
    print(importances.to_string())

    joblib.dump({"model": model, "features": features, "domain": args.domain}, args.out)
    print(f"\nModel saved to {args.out}")


if __name__ == "__main__":
    main()

