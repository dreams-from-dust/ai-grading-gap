"""
Computes the final comparison metrics reported in the paper's Table 1:
Quadratic Weighted Kappa, Pearson correlation, MAE, mean gap, and the
score-compression ratio, for all three graders across both domains.

Usage:
    python3 06_compute_final_metrics.py

Expects the six grading CSVs (ai_grades_essay_llama.csv, ai_grades_essay_qwen.csv,
claude_grades_essay.csv, ai_grades_shortanswer_llama.csv,
ai_grades_shortanswer_qwen.csv, claude_grades_shortanswer.csv) in the same folder.

Requirements:
    pip install pandas scikit-learn --break-system-packages
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

FILES = {
    ('essay', 'llama'): 'ai_grades_essay_llama.csv',
    ('essay', 'qwen'): 'ai_grades_essay_qwen.csv',
    ('essay', 'claude'): 'claude_grades_essay.csv',
    ('shortanswer', 'llama'): 'ai_grades_shortanswer_llama.csv',
    ('shortanswer', 'qwen'): 'ai_grades_shortanswer_qwen.csv',
    ('shortanswer', 'claude'): 'claude_grades_shortanswer.csv',
}


def main():
    rows = []
    for (domain, model), f in FILES.items():
        df = pd.read_csv(f)
        human = df['human_score'].round().astype(int)
        ai = df['ai_score'].round().astype(int)

        qwk = cohen_kappa_score(human, ai, weights='quadratic')
        gap = df['ai_score'] - df['human_score']
        mae = gap.abs().mean()
        pearson = df['human_score'].corr(df['ai_score'])
        human_std = df['human_score'].std()
        ai_std = df['ai_score'].std()
        compression_ratio = ai_std / human_std

        rows.append({
            'domain': domain, 'model': model, 'n': len(df),
            'QWK': round(qwk, 3), 'Pearson_r': round(pearson, 3), 'MAE': round(mae, 3),
            'mean_gap': round(gap.mean(), 3),
            'human_std': round(human_std, 3), 'ai_std': round(ai_std, 3),
            'compression_ratio(AI_std/Human_std)': round(compression_ratio, 3)
        })

    res = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(res.to_string(index=False))
    res.to_csv('final_metrics_summary.csv', index=False)
    print('\nSaved to final_metrics_summary.csv')


if __name__ == '__main__':
    main()
