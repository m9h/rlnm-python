"""Self-contained reports for annotation and experimental prediction workflows."""

import base64
import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _report(output, title, summary, figures):
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{max-width:1000px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;
background:#101827;color:#e3edf6}}img{{max-width:100%;background:white}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}
section{{padding:20px 0;border-top:1px solid #586779}}
</style><h1>{html.escape(title)}</h1>
<p>{html.escape(summary["interpretation"])}</p>"""
    for filename, caption in figures:
        encoded = base64.b64encode((output / filename).read_bytes()).decode()
        content += (
            f'<section><h2>{html.escape(caption)}</h2><img alt="{html.escape(caption)}" '
            f'src="data:image/png;base64,{encoded}"></section>'
        )
    content += "<section><h2>Results and settings</h2><pre>"
    content += html.escape(json.dumps(summary, indent=2, allow_nan=False))
    content += "</pre></section><p>Input/output hashes and versions: provenance.json</p></html>"
    (output / "report.html").write_text(content, encoding="utf-8")


def annotation_report(output, summary, diagnostics):
    rows = summary["results"]
    fig, ax = plt.subplots(figsize=(9, max(3, len(rows) * 0.45)))
    ax.barh(np.arange(len(rows)), [row["r"] for row in rows], color="#34789c")
    ax.set_yticks(np.arange(len(rows)), [r["annotation"] for r in rows])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson r (all tested annotations; see summary for spatial p and BH q)")
    fig.tight_layout()
    fig.savefig(output / "associations.png", dpi=120)
    plt.close(fig)
    figures = [("associations.png", "Spatial associations")]
    for number, item in enumerate(diagnostics):
        fig, axes = plt.subplots(
            1, len(item["blocks"]), figsize=(6 * len(item["blocks"]), 3.5), squeeze=False
        )
        for ax, block in zip(axes[0], item["blocks"], strict=True):
            x = block["distance"]
            ax.fill_between(x, block["surrogate_025"], block["surrogate_975"], alpha=0.25)
            ax.plot(x, block["surrogate_mean"], label="Null mean")
            ax.plot(x, block["empirical"], label="Observed", color="black")
            ax.set_title(f"Block {block['block']}; relative RMSE {block['relative_rmse']:.3f}")
            ax.set_xlabel("Distance (input units)")
            ax.set_ylabel("Smoothed semivariance")
            ax.legend()
        fig.tight_layout()
        filename = f"variogram-{number + 1:03d}.png"
        fig.savefig(output / filename, dpi=120)
        plt.close(fig)
        figures.append((filename, f"{item['annotation']}: variogram fit (review required)"))
    _report(output, "rLNM molecular annotation", summary, figures)


def benchmark_report(output, summary, rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter([r["baseline_mse"] for r in rows], [r["model_mse"] for r in rows])
    bound = max(max(r["baseline_mse"], r["model_mse"]) for r in rows)
    axes[0].plot([0, bound], [0, bound], "--", color="gray")
    axes[0].set_xlabel("Training atlas baseline MSE")
    axes[0].set_ylabel("Held-out model MSE")
    axes[1].bar(np.arange(len(rows)), [r["mse_improvement"] for r in rows])
    axes[1].axhline(0, color="gray")
    axes[1].set_xlabel("Held-out participant index")
    axes[1].set_ylabel("Baseline MSE − model MSE (positive is better)")
    fig.tight_layout()
    fig.savefig(output / "benchmark.png", dpi=120)
    plt.close(fig)
    _report(
        output,
        "Experimental EigenBrain-style benchmark",
        summary,
        [("benchmark.png", "Nested leave-one-participant-out predictions")],
    )
