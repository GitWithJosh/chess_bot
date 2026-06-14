import chess
import numpy as np
import keras
from keras import layers
#import tensorflow as tf


from reinforcement_learning.monte_carlo_tree_search.mcts_v2 import MCTS
from reinforcement_learning.monte_carlo_tree_search.nodes_and_edges_v2 import Node
from reinforcement_learning.helpers.converter import Converter


class BigNetwork:
    """
    Neural network architecture inspired by Leela Chess Zero / AlphaZero.

    Building blocks are Squeeze-and-Excitation (SE) residual blocks instead of
    plain ResNet blocks.  A typical configuration is 20 blocks with 256 filters.

    Input
    -----
    Shape: (8, 8, 112)  — channels-last, matching TensorFlow/Keras defaults.
    The 112 planes encode the board history used by AlphaZero/Leela.

    Residual tower
    --------------
    One initial Conv → BN → ReLU projects the input to `num_filters` channels.
    Each SE block contains:
        Conv(3x3) → BN → ReLU
        Conv(3x3) → BN
        SE attention: GlobalAvgPool → Dense(C/se_ratio, relu) → Dense(C, sigmoid)
                      → channel-wise scale of the second conv output
        Add(shortcut) → ReLU

    Policy head
    -----------
    Two Conv layers (no activation), Flatten and a final Dense producing
    `num_moves` logits (default 1858). Softmax activation on the final Dense.

    Value head  (WDL — Win / Draw / Loss)
    --------------------------------------
    Conv(1x1) → BN → ReLU → Flatten → Dense(num_filters, relu) → Dense(3, softmax).
    The three outputs are the probabilities for winning, drawing, and losing.
    Loss: categorical cross-entropy against the actual game result one-hot vector.
    """

    def __init__(
        self,
        input_shape: tuple = (8, 8, 112),
        num_res_blocks: int = 10,
        num_filters: int = 256,
        num_moves: int = 1858,
        se_ratio: int = 16,
        learning_rate: float = 0.001,
    ):
        self.input_shape = input_shape
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.num_moves = num_moves
        self.se_ratio = se_ratio
        self.learning_rate = learning_rate

        # self._infer = tf.function(
        #     lambda x: self.model(x, training=False),
        #     reduce_retracing=True,
        # )

        self.model = self._build()
        self._compile()

    # ------------------------------------------------------------------
    # Block builders
    # ------------------------------------------------------------------

    def _build_se_block(self, x, block_name: str):
        """
        Squeeze-and-Excitation residual block.

        Architecture:
            shortcut = x
            x = Conv(3x3) → BN → ReLU
            x = Conv(3x3) → BN
            # SE attention on x (before adding shortcut)
            se = GlobalAvgPool(x)
            se = Dense(C // se_ratio, relu)(se)
            se = Dense(C, sigmoid)(se)          # per-channel scale in [0, 1]
            x  = x * se                         # channel-wise rescaling
            x  = Add([shortcut, x])
            x  = ReLU(x)
        """
        C = self.num_filters
        shortcut = x

        # --- First conv ---
        x = layers.Conv2D(
            C, 3, padding="same", use_bias=False, name=f"{block_name}_conv1"
        )(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
        x = layers.ReLU(name=f"{block_name}_relu1")(x)

        # --- Second conv (no activation yet) ---
        x = layers.Conv2D(
            C, 3, padding="same", use_bias=False, name=f"{block_name}_conv2"
        )(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)

        # --- Squeeze: global average pooling → (batch, C) ---
        se = layers.GlobalAveragePooling2D(name=f"{block_name}_se_squeeze")(x)

        # --- Excitation: two FC layers ---
        se = layers.Dense(
            max(1, C // self.se_ratio),
            activation="relu",
            use_bias=True,
            name=f"{block_name}_se_fc1",
        )(se)
        se = layers.Dense(
            C,
            activation="sigmoid",
            use_bias=True,
            name=f"{block_name}_se_fc2",
        )(se)

        # --- Reshape to (batch, 1, 1, C) for broadcasting ---
        se = layers.Reshape((1, 1, C), name=f"{block_name}_se_reshape")(se)

        # --- Channel-wise scale + residual add ---
        x = layers.Multiply(name=f"{block_name}_se_scale")([x, se])
        x = layers.Add(name=f"{block_name}_add")([shortcut, x])
        x = layers.ReLU(name=f"{block_name}_relu2")(x)

        return x

    def _build_policy_head(self, x):
        """
        Policy head — outputs move probabilities over `num_moves` moves.

        Two Conv layers (the second projects to several planes), then Flatten
        and a final Dense to produce the `num_moves`-dimensional distribution.
        """
        x = layers.Conv2D(
            self.num_filters, 3, padding="same", use_bias=False, name="policy_conv1"
        )(x)
        x = layers.Conv2D(
            73,  # 73 output planes matching the move encoding
            1,
            padding="same",
            use_bias=False,
            name="policy_conv2",
        )(x)
        x = layers.Flatten(name="policy_flatten")(x)
        x = layers.Dense(self.num_moves, activation="softmax", name="policy_output")(x)
        return x

    def _build_value_head(self, x):
        """
        Value head — outputs WDL probabilities (Win, Draw, Loss).

        Single 1x1 conv for channel reduction, then two FC layers.
        Softmax over three classes instead of a scalar tanh.
        """
        x = layers.Conv2D(1, 1, use_bias=False, name="value_conv")(x)
        x = layers.BatchNormalization(name="value_bn")(x)
        x = layers.ReLU(name="value_relu")(x)
        x = layers.Flatten(name="value_flatten")(x)
        x = layers.Dense(self.num_filters, activation="relu", name="value_dense")(x)
        x = layers.Dense(3, activation="softmax", name="value_output")(x)
        return x

    # ------------------------------------------------------------------
    # Full model
    # ------------------------------------------------------------------

    def _build(self) -> keras.Model:
        inputs = layers.Input(shape=self.input_shape, name="board_input")

        # Initial projection conv
        x = layers.Conv2D(
            self.num_filters, 3, padding="same", use_bias=False, name="initial_conv"
        )(inputs)
        x = layers.BatchNormalization(name="initial_bn")(x)
        x = layers.ReLU(name="initial_relu")(x)

        # SE residual tower
        for i in range(self.num_res_blocks):
            x = self._build_se_block(x, block_name=f"se_block_{i}")

        # Heads
        policy_output = self._build_policy_head(x)
        value_output = self._build_value_head(x)

        return keras.Model(
            inputs=inputs,
            outputs=[policy_output, value_output],
            name="leela_network",
        )

    def _compile(self):
        """
        Policy loss : categorical cross-entropy vs. MCTS visit distribution.
        Value loss  : categorical cross-entropy vs. WDL one-hot game result.
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                "policy_output": "categorical_crossentropy",
                "value_output": "categorical_crossentropy",  # changed from MSE
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, board_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a single board position.

        Args:
            board_tensor: numpy array of shape (8, 8, 112)

        Returns:
            Tuple of (move_probabilities, wdl_probabilities)
                move_probabilities : numpy array of shape (num_moves,)
                wdl_probabilities  : numpy array of shape (3,)  — [win, draw, loss]
        """
        input_batch = np.expand_dims(board_tensor, axis=0).astype(np.float32)
        policy, value = self.model(input_batch, training=False)
        return np.asarray(policy[0]), np.asarray(value[0])
    
    def search_for_best_move(self, board: chess.Board, num_simulations: int) -> chess.Move:
        """
        Select the best move for the given position using MCTS search.

        Args:
            board: The current chess board position
            num_simulations: Number of MCTS simulations to run

        Returns:
            The best move according to MCTS (greedy, most-visited)
        """
        converter = Converter()
        mcts = MCTS(
            network=self,
            converter=converter,
            num_simulations=num_simulations,
        )
        root = Node(board)
        root = mcts.search(root, add_noise=False)
        return mcts.get_best_move(root, temperature=0)
    def outcome_to_wdl(self, outcomes: np.ndarray) -> np.ndarray:
        """
        Converts scalar game outcomes to WDL one-hot vectors.

        Input:  (batch_size, 1) or (batch_size,)  with values -1, 0, 1
        Output: (batch_size, 3)  — columns: [win=1, draw=0, loss=-1]
        """
        outcomes = outcomes.flatten()
        wdl = np.zeros((len(outcomes), 3), dtype=np.float32)
        wdl[outcomes == 1, 0] = 1.0  # Win
        wdl[outcomes == 0, 1] = 1.0  # Draw
        wdl[outcomes == -1, 2] = 1.0  # Loss
        return wdl

    def train(
        self,
        board_tensors: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        **kwargs,
    ):
        """
        Train the network on a batch of positions.

        Args:
            board_tensors  : numpy array of shape (batch_size, 8, 8, 112)
            policy_targets : numpy array of shape (batch_size, num_moves)
                             MCTS visit-count distributions (sum to 1 per row).
            value_targets  : numpy array of shape (batch_size, 3)
                             One-hot WDL vectors, e.g. [1,0,0] for a win.
            **kwargs       : forwarded to model.fit (epochs, batch_size, …)
        """
        return self.model.fit(
            board_tensors,
            {
                "policy_output": policy_targets,
                "value_output": self.outcome_to_wdl(value_targets),
            },
            **kwargs,
        )

    def save(self, path: str):
        """Save model weights to disk."""
        self.model.save_weights(path)

    def load(self, path: str):
        """Load model weights from disk."""
        self.model.load_weights(path)

    def summary(self):
        """Print model architecture summary."""
        self.model.summary()

    # make random weight and save under weights/random_model.weights.h5


if __name__ == "__main__":
    net = BigNetwork()
    net.save("reinforcement_learning/networks/weights/random_model.weights.h5")
    print(
        "Saved random weights for BigNetwork to reinforcement_learning/networks/weights/random_model.weights.h5"
    )