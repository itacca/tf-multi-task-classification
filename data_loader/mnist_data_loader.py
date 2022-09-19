import tensorflow as tf

from base_data_loader import DataLoader
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow import keras
from sklearn.model_selection import train_test_split


class MNISTDataLoader(DataLoader):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.load_data()

    def build_data_pipeline(
            self,
            dataset: tf.data.Dataset,
            dataset_size: int
    ) -> tf.data.Dataset:
        # Shuffle the data with a buffer size equal to the length of the
        # dataset. This ensures good shuffling.
        dataset = dataset.shuffle(
            dataset_size,
            seed=self.config["data"]["random_seed"]
        )
        # Batch the training dataset
        dataset = dataset.batch(self.config["data"]["batch_size"])
        # The AUTOTUNE parameter tells TensorFlow to build the pipeline
        # and then optimize such that the CPU can budget time for each
        # of the parameters in the pipeline. In this example, prefetching
        # buffer sizes for train and validation dataset should be optimized.
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def load_data(self) -> None:

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Dimension correction - (28, 28) -> (28, 28, 1)
        # We want to provide one extra dimension for our one channel
        # (e.g. RGB images would have 3)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        ###############################################
        # Create artificial labels for the second task.
        ###############################################
        # Since the original dataset does not have labels for multiple
        # tasks, we create artificial set of labels for the second task.
        # The second set of labels (hence the task #2 itself) is quite
        # trivial: if the image represents an odd number, we would assign
        # it the label "1", otherwise the label would be "0".

        y_train_labels_task_2 = list(map(lambda x: x % 2, y_train))
        y_test_labels_task_2 = list(map(lambda x: x % 2, y_test))

        num_of_classes = self.config["data"]["num_of_classes"]
        # Convert class vectors to binary class matrices (one-hot encoding).
        y_train_labels_task_1 = keras.utils.to_categorical(
            y_train, num_classes=num_of_classes
        )
        y_test_labels_task_1 = keras.utils.to_categorical(
            y_test, num_classes=num_of_classes
        )

        # Train - validation split for images and
        # both sets of labels (one set per each task).
        (
            train_images, valid_images,
            train_labels_task_1, valid_labels_task_1,
            train_labels_task_2, valid_labels_task_2
        ) = train_test_split(
            x_train,
            y_train_labels_task_1,
            y_train_labels_task_2,
            test_size=self.config["data"]["test_size"],
            # Dataset stratification: it is desirable to split the dataset
            # into train and test sets in a way that preserves the same
            # proportion of examples in each class as observed in the
            # original dataset.
            stratify=y_train,
            random_state=self.config["data"]["random_seed"],
            shuffle=True
        )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            # The first element in this outer tuple represents images,
            # the first element of inner tuple represents labels for the
            # first task, while the second element of the inner tuple
            # represents the labels for the second task.
            (train_images, (train_labels_task_1, train_labels_task_2))
        )
        # Build pipeline for training set
        self.train_dataset = self.build_data_pipeline(
            train_dataset, len(train_images))

        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (valid_images, (valid_labels_task_1, valid_labels_task_2))
        )
        # Build pipeline for validation set
        self.validation_dataset = self.build_data_pipeline(
            validation_dataset, len(valid_images))
        # Build pipeline for test set
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (x_test, (y_test_labels_task_1, y_test_labels_task_2))
        )
        self.test_dataset = self.test_dataset.batch(1)

    def get_train_data(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_validation_data(self) -> tf.data.Dataset:
        return self.validation_dataset

    def get_test_data(self) -> tf.data.Dataset:
        return self.test_dataset
