# AI Grading Gap

Empirical evaluation of AI-assisted grading reliability across essays and short-answer responses using human-assigned grades as the reference.

## Overview

This project evaluates the agreement between three AI models and human-assigned grades across two question types:

- **1,783 essays**
- **646 short-answer responses**

The study evaluates AI-generated grades using multiple metrics, including:

- Quadratic Weighted Kappa (QWK)
- Pearson correlation
- Mean Absolute Error (MAE)
- Mean grading gap
- AI-to-human score standard deviation ratio

The repository contains the analysis code, grading results, trained models, comparison results, final metrics, and exploratory analysis used in the study.

## Evaluation Setup

Three AI models were evaluated across two assessment domains:

| Domain | Number of Responses | Models |
|---|---:|---|
| Essay | 1,783 | Llama, Qwen, Claude |
| Short Answer | 646 | Llama, Qwen, Claude |

Human-assigned grades were used as the reference for evaluating AI-generated grades.

## Final Results

| Domain | Model | n | QWK | Pearson r | MAE | Mean Gap | AI/Human Std. Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Essay | Llama | 1,783 | -0.028 | -0.052 | 3.616 | -2.337 | 2.306 |
| Essay | Qwen | 1,783 | 0.274 | 0.351 | 2.212 | -1.069 | 1.675 |
| Essay | Claude | 1,783 | 0.381 | 0.657 | 1.812 | -1.789 | 0.936 |
| Short Answer | Llama | 646 | 0.198 | 0.381 | 0.816 | -0.779 | 0.679 |
| Short Answer | Qwen | 646 | 0.342 | 0.482 | 0.526 | -0.350 | 0.540 |
| Short Answer | Claude | 646 | 0.657 | 0.663 | 0.334 | 0.090 | 1.052 |

## Key Finding

The results show that AI grading agreement varies across both **model** and **question type**.

For essays, QWK values range from **-0.028 to 0.381**, while for short answers they range from **0.198 to 0.657**.

This variation indicates that AI grading reliability is not consistent across different models and assessment types. The finding provides empirical motivation for reliability-aware approaches to AI-assisted grading.