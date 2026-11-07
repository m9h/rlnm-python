"""Self-contained HTML QC report; visualization is not a registration certificate."""

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402


def _render(path):
    image = nib.as_closest_canonical(nib.load(path))
    data = image.get_fdata()
    bound = max(float(np.max(np.abs(data))), 1e-12)
    active = np.flatnonzero(np.any(data != 0, axis=(0, 1)))
    if not len(active):
        active = np.arange(data.shape[2])
    slices = np.quantile(active, [0.25, 0.5, 0.75]).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), facecolor="#101827")
    for ax, index in zip(axes, slices, strict=True):
        ax.imshow(data[:, :, index].T, origin="lower", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_title(
            f"z = {nib.affines.apply_affine(image.affine, [0, 0, index])[2]:.1f}",
            color="white",
            fontsize=10,
        )
        ax.set_axis_off()
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode(), bound


def write_report(output, summary, maps):
    output = Path(output)
    cards = []
    for title, filename in maps:
        encoded, bound = _render(output / filename)
        cards.append(
            f"<section><h2>{html.escape(title)}</h2>"
            f'<img alt="{html.escape(title)} axial slices" '
            f'src="data:image/png;base64,{encoded}">'
            f"<p>Symmetric colour scale: ±{bound:.4g}. "
            "Neurological orientation: image right is subject right. "
            "Empty maps remain visible.</p></section>"
        )
    content = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>rLNM analysis report</title><style>
body{max-width:1050px;margin:40px auto;padding:0 24px;background:#101827;color:#e3edf6;
font:16px/1.6 system-ui}h1{font-size:32px}h2{font-size:22px}section{margin:28px 0;
padding:22px;border:1px solid #344257;border-radius:12px}img{max-width:100%}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}p{color:#b6c6d8}
</style><h1>rLNM analysis report</h1>
<p>Grid checks passed for all inputs. This verifies file geometry, not anatomical
registration. Inspect original lesions and anatomy in your imaging viewer.</p>
<p>The report shows computed maps without an anatomical underlay. It does not
establish a causal mechanism, clinical prediction accuracy, or null calibration.</p>
"""
    content += (
        "<section><h2>Run summary</h2><pre>"
        + html.escape(json.dumps(summary, indent=2, allow_nan=False))
        + "</pre></section>"
    )
    content += "".join(cards) + "<p>Full settings and input hashes: provenance.json</p></html>"
    (output / "report.html").write_text(content)
