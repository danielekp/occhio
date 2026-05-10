"""Bar-only feature-norm visualization for ToyModel / ModelGrid."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel


def plot_bars(model_grid: ToyModel | ModelGrid):
    """Plot only the feature-norm bar charts for a 0D or 1D ModelGrid.

    Bar length is ``||W_i||^2`` per feature; bar color encodes total interference
    on a Viridis scale (dark = clean / monosemantic, yellow = polysemantic).

    Args:
        model_grid: A ToyModel or 0/1-dimensional ModelGrid.

    Returns:
        A plotly Figure with one bar chart per model in the grid.

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

    if model_grid.axes:
        axis = model_grid.axes[0]
        titles = [f"{axis.label} = {v:.3g}" for v in axis.values]
    else:
        titles = [""]

    fig = make_subplots(
        rows=1,
        cols=len(models),
        shared_yaxes=True,
        subplot_titles=titles,
        horizontal_spacing=0.01,
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
            row=1,
            col=k + 1,
        )
        fig.update_xaxes(
            range=[0, 1.2],
            tickmode="array",
            tickvals=[0, 1],
            showgrid=True,
            gridcolor="gray",
            gridwidth=0.5,
            griddash="dot",
            row=1,
            col=k + 1,
        )

    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        height=400,
        width=max(220, 180 * len(models)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=40, b=20),
    )
    return fig
