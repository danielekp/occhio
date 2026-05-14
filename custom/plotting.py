"""Bar-only feature-norm visualization for ToyModel / ModelGrid."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel


def plot_bars(
    model_grid: ToyModel | ModelGrid,
    *,
    width: int = 1400,
    row_height: int = 180,
    gif_path: str | Path | None = None,
    gif_duration: int = 400,
    gif_width: int = 1000,
    gif_height: int = 500,
):
    """Plot vertically-stacked feature-norm bar charts for a 0D or 1D ModelGrid.

    Bar length is ``||W_i||^2`` per feature; bar color encodes total interference
    on a Viridis scale (dark = clean / monosemantic, yellow = polysemantic).

    Args:
        model_grid: A ToyModel or 0/1-dimensional ModelGrid.
        width: Figure width in pixels.
        row_height: Height per stacked subplot in pixels.
        gif_path: If set, also render each snapshot as a frame and save an
            animated GIF to this path.
        gif_duration: Milliseconds per frame in the GIF.
        gif_width: Width of each GIF frame in pixels.
        gif_height: Height of each GIF frame in pixels.

    Returns:
        A plotly Figure with one bar chart per model, stacked vertically.

    Raises:
        ValueError: If model_grid is not 0 or 1-dimensional or is empty.
    """
    if isinstance(model_grid, ToyModel):
        model_grid = ModelGrid(
            create_model=lambda: model_grid,
            axes=[],
            broadcast_samples=False,
            _models=np.array([model_grid]),
        )

    if len(model_grid.shape) > 1:
        raise ValueError(
            f"plot_bars requires a 0 or 1-dimensional ModelGrid, "
            f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
        )

    models = list(model_grid.models.flat)
    if not models:
        raise ValueError("Cannot plot an empty ModelGrid.")

    axis = model_grid.axes[0] if model_grid.axes else None
    titles = (
        [f"{axis.label} = {v:.3g}" for v in axis.values]
        if axis is not None
        else [""]
    )

    fig = make_subplots(
        rows=len(models),
        cols=1,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=min(0.02, 1.0 / max(len(models) * 4, 1)),
    )

    for k, model in enumerate(models):
        norms = model.feature_norms.detach().cpu().numpy()
        interferences = model.total_feature_interferences.detach().cpu().numpy()
        is_last = k == len(models) - 1

        fig.add_trace(
            go.Bar(
                x=norms,
                y=list(reversed(range(len(norms)))),
                orientation="h",
                marker=dict(
                    color=interferences,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1.2,
                    showscale=is_last,
                    colorbar=dict(
                        title=dict(text="Interference", side="right", font=dict(size=10)),
                        thickness=10,
                    ),
                ),
                hovertemplate="||W<sub>%{y}</sub>||: %{x:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=k + 1,
            col=1,
        )
        fig.update_xaxes(
            range=[0, 1.2],
            tickmode="array",
            tickvals=[0, 1],
            showgrid=True,
            gridcolor="gray",
            gridwidth=0.5,
            griddash="dot",
            row=k + 1,
            col=1,
        )
        fig.update_yaxes(showticklabels=False, row=k + 1, col=1)

    fig.update_layout(
        height=row_height * len(models) + 40,
        width=width,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=40, b=20, l=40, r=40),
    )

    if gif_path is not None:
        _save_bars_gif(
            models,
            titles,
            Path(gif_path),
            duration=gif_duration,
            width=gif_width,
            height=gif_height,
        )

    return fig


def _save_bars_gif(
    models,
    titles,
    path: Path,
    *,
    duration: int,
    width: int,
    height: int,
):
    """Render each model as a single bar chart and stitch them into a GIF.

    Uses matplotlib (no browser dependency) instead of plotly's image export.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Saving GIFs requires matplotlib and Pillow. "
            "Install with: uv add --dev matplotlib Pillow"
        ) from exc

    n_features = len(models[0].feature_norms)
    cmap = cm.get_cmap("viridis")
    norm_color = colors.Normalize(vmin=0, vmax=1.2)
    dpi = 100
    figsize = (width / dpi, height / dpi)

    frames = []
    for model, title in zip(models, titles):
        norms = model.feature_norms.detach().cpu().numpy()
        interferences = model.total_feature_interferences.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        bar_colors = cmap(norm_color(interferences))
        positions = np.arange(n_features)[::-1]
        ax.barh(positions, norms, color=bar_colors, edgecolor="none")

        ax.set_xlim(0, 1.2)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([])
        ax.set_facecolor("#F0F0F0")
        ax.grid(axis="x", color="gray", linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=12)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        sm = cm.ScalarMappable(norm=norm_color, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Interference", fontsize=9)
        cbar.set_ticks([0, 1])

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
