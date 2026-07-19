"""Produce the study's report artifacts: metrics, CIs, calibration, writeup.

Everything the README's results section claims should be regenerable by
running this one stage. Outputs land in data/reports: a markdown summary,
calibration tables, and a reliability plot.
"""

from pathlib import Path

import pandas as pd

from pokertell.eval.metrics import bootstrap_delta_auc, calibration_table, single_report
from pokertell.models.train import loso_predictions


def dataset_summary(df: pd.DataFrame, target: str) -> dict:
    labeled = df[df[target].notna()]
    covered = labeled[
        (labeled.get("face_coverage", pd.Series(dtype=float)).fillna(0) >= 0.5)
        | (labeled.get("pose_coverage", pd.Series(dtype=float)).fillna(0) >= 0.5)
    ]
    return {
        "sessions": int(df["session_id"].nunique()),
        "hands": int(df["hand_id"].nunique()),
        "decisions": int(len(df)),
        "labeled": int(len(labeled)),
        "base_rate": float(labeled[target].mean()) if len(labeled) else float("nan"),
        "to_act_windows": int((labeled["window_source"] == "to_act").sum()),
        "behavior_covered": int(len(covered)),
    }


def per_player_breakdown(df: pd.DataFrame, target: str, min_n: int = 8) -> pd.DataFrame:
    labeled = df[df[target].notna()]
    rows = []
    for player, group in labeled.groupby("player"):
        if len(group) < min_n:
            continue
        rows.append(
            {
                "player": player,
                "n": len(group),
                "base_rate": round(float(group[target].mean()), 3),
                "aggr_rate": round(float(group["is_aggressive"].mean()), 3),
                "mean_time_to_act": round(float(group["time_to_act_s"].mean()), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False) if rows else pd.DataFrame()


def reliability_plot(tables: dict[str, list[dict]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="perfect")
    for name, rows in tables.items():
        if not rows:
            continue
        xs = [r["mean_predicted"] for r in rows]
        ys = [r["observed_rate"] for r in rows]
        ns = [r["count"] for r in rows]
        ax.plot(xs, ys, marker="o", label=name)
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(str(n), (x, y), fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed rate")
    ax.set_title("Reliability (bin counts annotated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_report(
    df: pd.DataFrame,
    target: str,
    base_cols: list[str],
    behavior_cols: list[str],
    out_dir: Path,
    model_kind: str = "logreg",
    min_coverage: float = 0.5,
    min_ablation_rows: int = 30,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled = df[df[target].notna()].copy()
    summary = dataset_summary(df, target)

    lines = [
        "# Results",
        "",
        f"Target: `{target}` | model: {model_kind} | evaluation: leave-one-session-out, "
        "pooled out-of-fold predictions.",
        "",
        "## Dataset",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in summary.items()]

    y, p_base, skipped = loso_predictions(labeled, target, base_cols, model_kind=model_kind)
    result: dict = {"summary": summary}
    tables: dict[str, list[dict]] = {}
    if len(y):
        base_metrics = single_report(y, p_base)
        result["baseline"] = base_metrics
        tables["betting baseline"] = calibration_table(y, p_base)
        lines += ["", "## Betting-only baseline", ""]
        lines += [f"- {k}: {round(v, 4) if isinstance(v, float) else v}"
                  for k, v in base_metrics.items()]
        if skipped:
            lines.append(f"- skipped sessions (single-class train fold): {skipped}")

    covered = labeled[
        (labeled.get("face_coverage", pd.Series(dtype=float)).fillna(0) >= min_coverage)
        | (labeled.get("pose_coverage", pd.Series(dtype=float)).fillna(0) >= min_coverage)
    ]
    cols = [c for c in behavior_cols if c in covered.columns]
    lines += ["", "## Ablation: baseline vs baseline + behavior", ""]
    if len(covered) >= min_ablation_rows and covered[target].nunique() == 2:
        yc, pc_base, _ = loso_predictions(covered, target, base_cols, model_kind=model_kind)
        yc2, pc_full, _ = loso_predictions(
            covered, target, base_cols + cols, model_kind=model_kind
        )
        if len(yc) and len(yc) == len(yc2):
            base_m = single_report(yc, pc_base)
            full_m = single_report(yc, pc_full)
            hand_groups = covered["hand_id"].to_numpy()[: len(yc)]
            ci = bootstrap_delta_auc(yc, pc_base, pc_full, groups=hand_groups)
            result["ablation"] = {"base": base_m, "full": full_m, "delta_ci": ci}
            tables["baseline (covered)"] = calibration_table(yc, pc_base)
            tables["with behavior"] = calibration_table(yc, pc_full)
            lines += [
                f"- rows with behavior coverage: {len(covered)}",
                f"- AUC baseline: {base_m['auc']:.4f}",
                f"- AUC with behavior: {full_m['auc']:.4f}",
                f"- delta AUC: {full_m['auc'] - base_m['auc']:+.4f} "
                f"(bootstrap 95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}], "
                f"grouped by hand)",
                f"- log loss: {base_m['logloss']:.4f} -> {full_m['logloss']:.4f}",
                "",
                "A delta whose CI includes zero is a null result and is reported as "
                "such. Context: the literature's ceilings put credible behavioral "
                "deltas at a few hundredths of AUC.",
            ]
    else:
        result["ablation"] = None
        lines += [
            f"- skipped: {len(covered)} behavior-covered rows "
            f"(minimum {min_ablation_rows}). Coverage grows with seat maps and "
            "footage, not with a lower bar.",
        ]

    breakdown = per_player_breakdown(df, target)
    if len(breakdown):
        lines += ["", "## Per-player (labeled decisions)", "", breakdown.to_markdown(index=False)]
        breakdown.to_csv(out_dir / "per_player.csv", index=False)

    if tables:
        reliability_plot(tables, out_dir / "reliability.png")
        lines += ["", "![reliability](reliability.png)"]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    result["report_path"] = str(report_path)
    return result
