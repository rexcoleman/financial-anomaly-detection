# Substack Intro — FP-04

This week I trained an ML fraud detector on 100K transactions and discovered that my CFA training might be worth more than my model architecture. CFA-informed rules alone score 0.898 AUC — 91% of what XGBoost achieves. The SHAP analysis confirms it: 8 of the top 20 predictive features come from domain expertise, not raw data.

I also applied controllability analysis to fraud detection for the first time — the 5th security domain where this methodology produces actionable results. System-controlled features retain 81% of detection capability even against a sophisticated fraudster.

Full write-up with code, SHAP rankings, and complexity curves below.
