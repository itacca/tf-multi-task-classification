from enum import Enum
from typing import Dict, Optional

from matplotlib import pyplot as plt


class PlotType(Enum):
    LOSS = "LOSS"
    ACCURACY = "ACCURACY"


def plot_learning_curves(
        history: Optional[Dict],
        plot_type: PlotType,
        window_title: str = "Learning curves"
) -> None:
    """Plot the learning curve for either loss or accuracy."""
    metric = str(plot_type.value).lower()

    window_title += f": {metric.title()}"
    fig = plt.figure()
    fig.canvas.manager.set_window_title(window_title)
    plt.xlabel("Epoch")
    plt.ylabel(metric.title())
    plt.plot(history["task_1_accuracy"], label="train_1")
    plt.plot(history["task_2_accuracy"], label="train_2")

    plt.plot(history["val_task_1_accuracy"], label="val_1")
    plt.plot(history["val_task_2_accuracy"], label="val_2")
    plt.legend()
    plt.show()
