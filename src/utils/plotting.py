"""Shared plotting defaults for the cleaned research project."""

import matplotlib.pyplot as plt


def apply_project_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
