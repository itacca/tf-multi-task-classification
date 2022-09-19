import os
import tensorflow as tf
from typing import Dict, Tuple

from loguru import logger

from base_data_loader import DataLoader
from base_model import BaseModel
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class MNISTModel(BaseModel):
    """MNIST dataset classification with modular building blocks."""

    def __init__(self, config: dict, data_loader: DataLoader) -> None:
        super().__init__(config, data_loader)

    def _apply_preprocessing(self, inputs: layers.Layer) -> layers.Layer:
        """Preprocess input data during the forward pass.

        The preprocessing includes rescaling and augmentation. """
        # Rescale inputs - scale images to [0, 1]
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
        x = self._apply_augmentation(x)
        return x

    def _apply_augmentation(self, inputs: layers.Layer) -> layers.Layer:
        """Apply basic augmentation layers to the provided input layer.

        The included augmentation layers perform random flipping, random
        rotation, random zoom and random translation. It should work well
        for the majority of downstream tasks, but for specific use cases,
        e.g. the ones whose inputs are not translation invariant, these
        augmentation layers should be skipped or changed."""
        augmentation_config = self.config["model"]["augmentation"]
        if not augmentation_config["active"]:
            logger.info("Skipping augmentations")
            return inputs
        data_augmentation = self.get_augmentations(augmentation_config)

        augmented_input = data_augmentation(inputs)
        logger.info("Augmentations applied.")

        return augmented_input

    @staticmethod
    def get_augmentations(augmentation_config: Dict) -> layers.Layer:
        """Return the augmentations stored in a Sequential layer.

        The method could be used while training to perform augmentation
        on the training dataset, or to visualize how the augmentation
        functions would change the original training dataset.
        """
        factors = augmentation_config["random_translation"]["factors"]
        height, width = factors[0], factors[1]
        data_augmentation = Sequential([
            layers.experimental.preprocessing.RandomFlip(
                augmentation_config["random_flip"]["mode"]
            ),
            layers.experimental.preprocessing.RandomRotation(
                augmentation_config["random_rotation"]["factor"]
            ),
            layers.experimental.preprocessing.RandomZoom(
                augmentation_config["random_zoom"]["height_factor"],
                augmentation_config["random_zoom"]["width_factor"]
            ),
            layers.experimental.preprocessing.RandomTranslation(
                height_factor=height, width_factor=width,
                fill_mode='reflect', interpolation='bilinear',
            )
        ])
        return data_augmentation

    def build(self) -> None:
        """Build simple CNN model for this peculiar problem. """
        input_shape = tuple(self.config["model"]["input"])
        inputs = keras.Input(shape=input_shape)

        x = self._apply_preprocessing(inputs)

        conv_layers_stack = self.config["model"]["stack"]
        for conv_layer in conv_layers_stack:
            x = layers.Conv2D(
                conv_layer["filters"],
                kernel_size=(3, 3),
                activation="relu",
                padding=conv_layer["padding"],
                strides=conv_layer["strides"],
                name=conv_layer["name"]
            )(x)
            # Note: there is an open discussion in Deep Learning
            # community whether Batch normalization layer should
            # be placed before or after activation function. Because
            # of the following explanation, this proposed architecture
            # has Batch normalization layer applied before activation
            # function.

            # Explanation: Take a look at the answer: (link to the
            # Stackoverflow comment: https://stackoverflow.com/
            # questions/34716454/where-do-i-call-the-batchnormalization-
            # function-in-keras#comment78826470_37979391 ).
            if conv_layer["bn_next"]:
                x = layers.BatchNormalization()(x)
            if conv_layer["relu_next"]:
                x = layers.ReLU()(x)

            if conv_layer["strides"] == 1:
                # We don't want to apply both pooling and
                # stride step of size 2 since it would reduce
                # the spatial size of the image significantly.
                x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        if self.config["model"]["closing_layer"] == "max_pooling":
            x = layers.GlobalMaxPooling2D()(x)
        elif self.config["model"]["closing_layer"] == "flatten":
            x = layers.Flatten()(x)
            x = layers.Dropout(0.7)(x)
        else:
            raise NotImplementedError(
                f"The provided choice {self.config['model']['closing_layer']}"
                f" is not supported / implemented."
            )
        # Multi-head classification
        x = layers.Dense(1024)(x)

        # 10 class classification
        # Here we try with two Fully-Connected layers.
        x_1 = layers.Dense(512)(x)
        x_1 = layers.Dense(256)(x_1)
        output_1 = layers.Dense(
            self.config["model"]["output"],
            activation="softmax",
            name="task_1"
        )(x_1)

        # 2 class classification
        # The assumption is that the binary classification is easier
        # task than the 10-class classification, so we put only one
        # Fully connected layer.
        x_2 = layers.Dense(256)(x)
        output_2 = layers.Dense(
            1, activation="sigmoid", name="task_2"
        )(x_2)
        self.model = keras.Model(inputs, [output_1, output_2])

    def train(self) -> dict:
        """Train the model on the provided data. """
        self.model.summary()

        if self.config["train"]["train_loop"] == "custom":
            history = self.custom_train_loop()
        elif self.config["train"]["train_loop"] == "classic":
            history = self.classic_train_loop()
        else:
            raise Exception("The specified train loop "
                            "approach does not exist! ")
        return history

    def classic_train_loop(self) -> Dict:
        """Model training using predefined 'compile' and 'fit' methods. """
        self.model.compile(
            # We don't need 'sparse' categorical CE function since our
            # labels are one-hot encoded for 10-class classification task.
            loss={
                "task_1": "categorical_crossentropy",
                "task_2": "binary_crossentropy"
            },
            optimizer=self.config["train"]["optimizer"],
            metrics=self.config["train"]["metrics"]
        )
        callbacks = self.initialize_callbacks()
        train_dataset = self.data_loader.get_train_data()
        validation_dataset = self.data_loader.get_validation_data()
        history = self.model.fit(
            train_dataset,
            batch_size=self.config["train"]["batch_size"],
            epochs=self.config["train"]["epochs"],
            validation_data=validation_dataset,
            callbacks=callbacks
        )
        return history.history

    def custom_train_loop(self) -> any:
        """Model training using 'GradientTape', custom training loop. """
        # Since we are not using 'compile' method, we initialize
        # optimizers, loss functions and metrics on our own.
        optimizer = tf.keras.optimizers.Adam()

        # Loss functions for 10-class and 2-class
        # classification, respectively.
        loss_function_1 = tf.keras.losses.CategoricalCrossentropy(
            name="task_1_loss")
        loss_function_2 = tf.keras.losses.BinaryCrossentropy(
            name="task_2_loss")

        # NOTE: TF is able to deduce the right accuracy metric
        # necessary for the specific task: e.g. in the case of 10 class
        # classification, the right one should be "CategoricalAccuracy".
        # However, when there are two learning tasks in the custom
        # training loop, each metric needs to be defined precisely.
        # Accuracy metrics for the first task.
        train_metrics_1 = tf.keras.metrics.CategoricalAccuracy(
            name="task_1_train_accuracy")
        val_metrics_1 = tf.keras.metrics.CategoricalAccuracy(
            name="task_1_val_accuracy")

        # Accuracy metrics for the second task.
        train_metrics_2 = tf.keras.metrics.BinaryAccuracy(
            name="task_2_train_accuracy")
        val_metrics_2 = tf.keras.metrics.BinaryAccuracy(
            name="task_2_val_accuracy")

        metrics = [
            train_metrics_1, train_metrics_2,
            val_metrics_1, val_metrics_2
        ]

        # Training history would be useful for plotting
        # training graphs once the training process is finished.
        # Initialize training history, epoch level.
        epoch_training_history = {
            loss_function_1.name: [],
            loss_function_2.name: []
        }
        for metric in metrics:
            epoch_training_history[metric.name] = []

        # Prepare train and validation datasets.
        train_dataset = self.data_loader.get_train_data()
        validation_dataset = self.data_loader.get_validation_data()

        epochs = self.config["train"]["epochs"]
        batch_size = self.config["train"]["batch_size"]
        loss_task_1 = 0
        loss_task_2 = 0

        # The following code actually replaces 'fit' method. This is
        # where the whole training procedure happens (validation as well).
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            for step, batch in enumerate(train_dataset):

                loss_task_1, loss_task_2 = train_step(
                    model=self.model,
                    batch=batch,
                    loss_function_1=loss_function_1,
                    loss_function_2=loss_function_2,
                    optimizer=optimizer,
                    train_metrics_1=train_metrics_1,
                    train_metrics_2=train_metrics_2
                )
                if step % 100 == 0:
                    logger.info(
                        f"Step {step}: training loss = "
                        f"{float(loss_task_1)}, {float(loss_task_2)}"
                    )
                    logger.info(
                        f"Seen so far: {(step + 1) * batch_size} samples"
                    )
            train_accuracy_1 = train_metrics_1.result()
            train_accuracy_2 = train_metrics_2.result()

            epoch_training_history[
                loss_function_1.name].append(float(loss_task_1))
            epoch_training_history[
                loss_function_2.name].append(float(loss_task_2))

            print(
                "Training accuracy over the epoch: %.4f, %.4f"
                % (float(train_accuracy_1), float(train_accuracy_2))
            )

            # Run a validation loop at the end of each epoch
            for (
                    x_batch_val, (y_batch_val_1, y_batch_val_2)
            ) in validation_dataset:
                # Forward pass
                val_output_1, val_output_2 = self.model(x_batch_val)

                val_metrics_1.update_state(y_batch_val_1, val_output_1)
                val_metrics_2.update_state(y_batch_val_2, val_output_2)
            val_accuracy_1 = val_metrics_1.result()
            val_accuracy_2 = val_metrics_2.result()

            for metric in metrics:
                epoch_training_history[
                    metric.name
                ].append(float(metric.result()))
                metric.reset_states()

            print(
                "Validation accuracy: %.4f, %.4f"
                % (float(val_accuracy_1), float(val_accuracy_2)))
        return epoch_training_history

    def initialize_callbacks(self):
        """Creates and retrieves callback functions for the upcoming training.

        Instantiated callback functions:
            1. Early stopping
            2. Model checkpointing
            3. Tensor board
        """

        # 'patience' defines how many epochs can pass without improvement
        # of desired metric (accuracy or loss) until the training is stopped
        # 'min_delta' specifies minimal amount of units (1 unit for mean
        # squared error, or 1 % for accuracy) that is considered improvement.
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        )
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(
                self.config["train"]["checkpoint_dir"],
                "checkpoint-{epoch:02d}-loss{val_loss:.3f}.h5"
            ),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        tensor_board = keras.callbacks.TensorBoard(log_dir="./logs")
        callbacks = [
            early_stopping_callback,
            model_checkpoint_callback,
            tensor_board
        ]
        return callbacks

    def evaluate(self) -> None:
        """Evaluate trained model on test data with corresponding labels."""
        if self.config["train"]["train_loop"] == "custom":
            self.custom_evaluate()
        elif self.config["train"]["train_loop"] == "classic":
            self.classic_evaluate()
        else:
            raise Exception("The specified train loop "
                            "approach does not exist! ")

    def classic_evaluate(self) -> None:
        """Evaluate model trained by classic train loop. """
        test_dataset = self.data_loader.get_test_data()
        score = self.model.evaluate(test_dataset, verbose=1)
        # Score[0] - total loss (in our case, sum of specific losses)
        # Score[1] - loss for task 1
        # Score[2] - loss for task 2
        # Score[3] - accuracy for task 1 (Categorical Accuracy)
        # Score[4] - accuracy for task 2 (Binary Accuracy)
        average_accuracy = (score[3] + score[4]) / 2
        print(f"Test loss: {score[0]}")
        print(f"Test accuracy: {average_accuracy}")

    def custom_evaluate(self) -> None:
        """Evaluate model trained by custom train loop. """
        test_dataset = self.data_loader.get_test_data()

        # Define losses like in the train loop.
        loss_function_1 = tf.keras.losses.CategoricalCrossentropy(
            name="task_1_loss")
        loss_function_2 = tf.keras.losses.BinaryCrossentropy(
            name="task_2_loss")

        # Define average loss.
        loss_avg = tf.keras.metrics.Mean()

        # Define task-specific accuracies.
        metric_1 = tf.keras.metrics.CategoricalAccuracy(
            name="val_task_1_accuracy")
        metric_2 = tf.keras.metrics.BinaryAccuracy(
            name="val_task_2_accuracy")

        for x_batch_val, (y_batch_val_1, y_batch_val_2) in test_dataset:
            # Forward pass
            val_output_1, val_output_2 = self.model(x_batch_val)

            loss_task_1 = loss_function_1(y_batch_val_1, val_output_1)
            loss_task_2 = loss_function_2(y_batch_val_2, val_output_2)

            # We treat both losses equally - weights are the same
            # total_loss = loss_task_1 + loss_task_2
            total_loss = [loss_task_1, loss_task_2]
            loss_avg.update_state(total_loss)

            metric_1.update_state(y_batch_val_1, val_output_1)
            metric_2.update_state(y_batch_val_2, val_output_2)
        accuracy_task_1 = metric_1.result()
        accuracy_task_2 = metric_2.result()
        # NOTE: There different ways to incorporate the separate
        # accuracies in one final accuracy, this is the easiest
        # way to implement one of it.
        average_accuracy = (accuracy_task_1 + accuracy_task_2) / 2
        print(f"Test loss: {loss_avg.result()}")
        print(f"Test accuracy: {average_accuracy}")

    def predict(self):
        pass


# NOTE 1: 'tf.function' annotation actually helps this part of code to
# run much faster. It is not possible to place this method inside a custom
# class like the upper one - unless the inherited class is 'tf.keras.Model'.
# NOTE 2: In order to replicate accuracies obtained by the classic 'fit'
# method, it seems as a necessity to have 'tf.function' annotation on the
# custom train loop. For some reason, it stabilizes the network training,
# helping it converge earlier.
@tf.function
def train_step(
        model: tf.keras.Model,
        batch: Tuple,
        loss_function_1: tf.keras.losses.Loss,
        loss_function_2: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        train_metrics_1: tf.keras.metrics.Metric,
        train_metrics_2: tf.keras.metrics.Metric
) -> Tuple:
    """Custom train loop.

    The function encapsulates forward pass, loss calculation, gradient
    optimization and logging of the metrics.
    """
    # 'GradientTape' keeps track of trainable variables - its method
    # 'gradient' calculates the gradient update for each trainable
    # variable, based on the calculated loss.
    with tf.GradientTape() as tape:
        # The number of elements in the tuple with labels is equal
        # to the number of tasks that the model is being trained on.
        # In our case, 2.
        x_batch, (y_batch_1, y_batch_2) = batch

        # Forward pass
        output_1, output_2 = model(x_batch, training=True)

        # Loss for 10 class classification task.
        loss_task_1 = loss_function_1(y_batch_1, output_1)
        # Loss for 2 class classification task.
        loss_task_2 = loss_function_2(y_batch_2, output_2)

        # We treat both losses equally - weights are the same (1).
        total_loss = [loss_task_1, loss_task_2]
    gradients = tape.gradient(
        total_loss, model.trainable_weights
    )
    optimizer.apply_gradients(
        zip(gradients, model.trainable_weights)
    )

    # Update task-related metrics.
    train_metrics_1.update_state(y_batch_1, output_1)
    train_metrics_2.update_state(y_batch_2, output_2)

    return loss_task_1, loss_task_2
