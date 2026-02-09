from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def animate_position_plotly(t: np.ndarray, y: np.ndarray, max_frames: int = 200) -> None:
    """Interactive Plotly animation of position vs time."""
    idx = np.linspace(0, len(t) - 1, min(max_frames, len(t))).astype(int)

    frames = [
        go.Frame(
            data=[
                go.Scatter(x=t, y=y, mode="lines", name="Position"),
                go.Scatter(
                    x=[t[i]],
                    y=[y[i]],
                    mode="markers",
                    name="Current",
                    marker=dict(size=10),
                ),
            ],
            name=str(i),
        )
        for i in idx
    ]

    fig = go.Figure(
        data=[
            go.Scatter(x=t, y=y, mode="lines", name="Position"),
            go.Scatter(
                x=[t[idx[0]]],
                y=[y[idx[0]]],
                mode="markers",
                name="Current",
                marker=dict(size=10),
            ),
        ],
        frames=frames,
    )

    fig.add_hline(y=0.0, line_dash="dash")

    fig.update_layout(
        title="Position Over Time",
        xaxis_title="Time",
        yaxis_title="Position",
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 30, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[f.name], {"mode": "immediate"}],
                        label=f"{t[int(f.name)]:.2f}",
                    )
                    for f in frames
                ],
                currentvalue={"prefix": "t = "},
            )
        ],
    )

    fig.show()
