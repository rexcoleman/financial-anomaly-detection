# LinkedIn Native Post — FP-04

I trained XGBoost on 100K financial transactions. AUC 0.987. But the most interesting finding? CFA-informed rule-based scoring hits 0.898 on its own.

Domain expertise captures 91% of what ML captures. Here's what that means:

- **8 of the top 20 SHAP features are CFA-informed.** Amount-to-median ratios, merchant risk tiers, suspicious timing patterns. Features a pure ML practitioner wouldn't engineer.
- **Rules aren't dead.** A well-designed rule baseline provides a 0.898 AUC floor with zero ML complexity. ML adds the last 9 percentage points by finding non-linear interactions.
- **Controllability analysis quantifies adversarial robustness.** Using only system-controlled features (card BIN, device fingerprint, address verification), the model retains 81% of detection capability — even if a fraudster perfectly manipulates everything they control.
- **Shallow trees beat deep ones.** Complexity curves show XGBoost peaks at max_depth=2. Deeper trees overfit without improving generalization.
- **This is the 5th domain where controllability analysis works.** Network security, CVE prediction, AI agents, crypto migration, and now fraud. One methodology, five domains, same actionable insight.

The takeaway for security teams: before you ship a model, ask "what's the detection floor from features the attacker can't control?" That number is what your CISO actually needs.

Full analysis: [link]

Code: github.com/rexcoleman/financial-anomaly-detection

---
Link in first comment
