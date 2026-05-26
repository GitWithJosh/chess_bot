import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from reinforcement_learning.helpers.converter import Converter
import chess

class SmallerNetwork:
    """
        Args:
        input_shape: Board tensor shape (8, 8, 112)
        num_res_blocks: Number of residual blocks
        num_filters: Number of convolutional filters per layer
        num_moves: Number of possible moves (1858 for chess)
        learning_rate: Learning rate for Adam optimizer
    """
 
    def __init__(
        self,
        input_shape: tuple = (8, 8, 112),
        num_res_blocks: int = 5,
        num_filters: int = 128,
        num_moves: int = 1858,
        learning_rate: float = 0.001,
    ):
        self.input_shape = input_shape
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.num_moves = num_moves
        self.learning_rate = learning_rate

        self.model = self._build()
        self._compile()

    def _build_residual_block(self, x, block_name: str):
        """Single residual block with two conv layers and skip connection."""
        shortcut = x

        x = layers.Conv2D(self.num_filters, 3, padding="same", use_bias=False, name=f"{block_name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
        x = layers.ReLU(name=f"{block_name}_relu1")(x)

        x = layers.Conv2D(self.num_filters, 3, padding="same", use_bias=False, name=f"{block_name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)

        x = layers.Add(name=f"{block_name}_add")([shortcut, x])
        x = layers.ReLU(name=f"{block_name}_relu2")(x)

        return x

    def _build_policy_head(self, x):
        """Policy head: outputs move probabilities."""
        x = layers.Conv2D(2, 1, use_bias=False, name="policy_conv")(x)
        x = layers.BatchNormalization(name="policy_bn")(x)
        x = layers.ReLU(name="policy_relu")(x)
        x = layers.Flatten(name="policy_flatten")(x)
        x = layers.Dense(self.num_moves, activation="softmax", name="policy_output")(x)
        return x

    def _build_value_head(self, x):
        """Value head: outputs position evaluation between -1 and 1."""
        x = layers.Conv2D(1, 1, use_bias=False, name="value_conv")(x)
        x = layers.BatchNormalization(name="value_bn")(x)
        x = layers.ReLU(name="value_relu")(x)
        x = layers.Flatten(name="value_flatten")(x)
        x = layers.Dense(self.num_filters, activation="relu", name="value_dense")(x)
        x = layers.Dense(1, activation="tanh", name="value_output")(x)
        return x

    def _build(self) -> keras.Model:
        """Build the full network architecture."""
        inputs = layers.Input(shape=self.input_shape, name="board_input")

        # Initial convolution to project input channels to filter size
        x = layers.Conv2D(self.num_filters, 3, padding="same", use_bias=False, name="initial_conv")(inputs)
        x = layers.BatchNormalization(name="initial_bn")(x)
        x = layers.ReLU(name="initial_relu")(x)

        # Residual tower
        for i in range(self.num_res_blocks):
            x = self._build_residual_block(x, block_name=f"res_block_{i}")

        # Two output heads
        policy_output = self._build_policy_head(x)
        value_output = self._build_value_head(x)

        return keras.Model(
            inputs=inputs,
            outputs=[policy_output, value_output],
            name="chess_network",
        )

    def _compile(self):
        """Compile the model with appropriate losses for each head.

        Policy head: categorical crossentropy (target = MCTS visit distribution)
        Value head: mean squared error (target = game outcome: -1, 0, or 1)
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                "policy_output": "categorical_crossentropy",
                "value_output": "mean_squared_error",
            },
        )

    def predict(self, board_tensor: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference on a single board position.

        Args:
            board_tensor: numpy array of shape (8, 8, 112)

        Returns:
            Tuple of (move_probabilities, position_evaluation)
                move_probabilities: numpy array of shape (1858,)
                position_evaluation: float between -1 and 1
        """
        input_batch = np.expand_dims(board_tensor, axis=0)
        policy, value = self.model.predict(input_batch, verbose=0)
        return policy[0], value[0][0]

    def train(self, board_tensors: np.ndarray, policy_targets: np.ndarray, value_targets: np.ndarray, **kwargs):
        """Train the network on a batch of positions.

        Args:
            board_tensors: numpy array of shape (batch_size, 8, 8, 112)
            policy_targets: numpy array of shape (batch_size, 1858) — MCTS visit distributions
            value_targets: numpy array of shape (batch_size, 1) — game outcomes (-1, 0, or 1)
            **kwargs: additional arguments passed to model.fit (epochs, batch_size, etc.)
        """
        return self.model.fit(
            board_tensors,
            {"policy_output": policy_targets, "value_output": value_targets},
            **kwargs,
        )

    def save(self, path: str):
        """Save the model weights to disk."""
        self.model.save_weights(path)

    def load(self, path: str):
        """Load model weights from disk."""
        self.model.load_weights(path)

    def summary(self):
        """Print model architecture summary."""
        self.model.summary()