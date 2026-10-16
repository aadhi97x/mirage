"""
Training entry point. Loads the track's dataset files from ./data, trains the three scikit-learn models from scratch
(no pretrained weights, no external API) and writes metrics and sample outputs to docs/sample_outputs/.

    python train.py

The same code runs inside the app (ml.run_all); this script just makes the training reproducible outside the UI.
"""
from __future__ import annotations
import json
import pathlib
import time

import analysis as A
import ml

OUT = pathlib.Path(__file__).parent / "docs" / "sample_outputs"


def main() -> None:
    t0 = time.time()
    raw, o = A.load_all()
    print(f"Loaded {len(raw['transactions']):,} orders, {len(raw['products'])} products, {len(raw['campaigns']):,} campaigns from {A.DATA_DIR}")
    res = ml.run_all(raw, o)
    b, an, u = res["baseline"], res["anomalies"], res["uplift"]

    OUT.mkdir(parents=True, exist_ok=True)
    b["table"].to_csv(OUT / "weekly_baseline_table.csv", index=False)
    an["all"].to_csv(OUT / "product_anomaly_scores.csv", index=False)
    report = {
        "seed": ml.SEED,
        "baseline_gradient_boosting": {k: v for k, v in b.items() if k != "table"},
        "isolation_forest": {"products_scored": an["n_products"], "top_10": an["table"][["product_id", "anomaly_score", "most_unusual_on", "direction"]].to_dict("records")},
        "uplift_t_learner": u,
    }
    (OUT / "model_report.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"\n1) Baseline (gradient boosting): trained on {b['train_weeks']} weeks, tested on {b['test_weeks']}")
    print(f"   error {b['mae_model']:.1f} orders/week vs {b['mae_naive']:.1f} for a plain average; R2 {b['r2']:.2f}")
    print(f"   spend importance {b['spend_importance']:+.2f} +/- {b['spend_importance_sd']:.2f} -> {'model beats average' if b['beats_naive'] else 'no better than average'}")
    print(f"\n2) Isolation Forest: {an['n_products']} products scored; most unusual:")
    print(an["table"][["product_id", "anomaly_score", "most_unusual_on", "direction"]].head(5).to_string(index=False))
    print(f"\n3) Uplift (T-learner): {u['mean_uplift']:+.3f} units/order from a 15-25% discount (95% interval {u['ci'][0]:+.3f} to {u['ci'][1]:+.3f}); "
          f"baseline {u['control_mean']:.2f}")
    print(f"\nWrote {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
