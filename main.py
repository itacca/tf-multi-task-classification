"""Multi-task classification on MNIST dataset, using TF 2.2+.

This script allows training and evaluating the CNN architecture on two
classification tasks:
1. MNIST 10 class classification
2. MNIST 2 class classification

The first task is the classic MNIST classification, while the second one
requires from the model to learn odd and even representation of numbers
present in the image, and classify them to the respective classes (for
 the sake of simplicity, they are denoted as "0" and "1").

All steps in the pipeline (data loading / visualisation / augmentation,
model build, model training and evaluation) are implemented as separate
modules, following the OOP concept.

By running this script, the full pipeline would be run:
1. Dataset loading, batching and prefetching using 'tf.data' Dataset
2. Dataset visualisation: inspection of both original samples from the
    dataset, and the images after the augmentation layer is applied.
3. Model build-up: create the custom architecture specified in the
    configuration file. All parameters (number of ConvLayers,
    existence of Batch Normalization and Pooling layers, etc.) could
    be specified via configuration file.
4. Model training: the whole process is supported by logging tool -
    MLflow, so we are able to track performance across individual
    experiments (where each experiment is denoted with one set of
    the hyper-parameters).
5. Model evaluation on the test dataset: simple accuracy metric.
"""
import os

import mlflow

from json_handling import load_json, save_json
from mnist_data_loader import MNISTDataLoader
from mnist_model import MNISTModel
from plotting import plot_learning_curves, PlotType
from tf_batch_visualisation import visualise_dataset_sample

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

# Note: this line solves a problem with CuDNN
# The origin of this issue could be missmatch between versions
# of CuDNN and TF. It is only reflected when using CNN layers.
# Log from TF:
# ' Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH
#   environment variable is set. Original config value was 0. '
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

CONFIG_PATH: str = "config_files/config.json"


def main() -> None:
    # Load configuration file
    config = load_json(CONFIG_PATH)
    mnist_data_loader = MNISTDataLoader(config)

    # Visualise training dataset (original & augmented images)
    augmentation_config = config["model"]["augmentation"]
    visualise_dataset_sample(
        data_loader=mnist_data_loader,
        augmentation_config=augmentation_config
    )

    mnist_model = MNISTModel(config, mnist_data_loader)
    mnist_model.build()

    history = mnist_model.train()
    mnist_model.evaluate()
    save_json(history, config["train"]["history_save_path"])

    plot_learning_curves(history, PlotType.ACCURACY)


if __name__ == '__main__':
    main()
