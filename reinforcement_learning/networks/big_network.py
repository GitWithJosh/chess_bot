import chess
import numpy as np
import tensorflow as tf
import keras
from keras import layers

from monte_carlo_tree_search.mcts_v2 import MCTS
from monte_carlo_tree_search.nodes_and_edges_v2 import Node
from helpers.converter import Converter


class BigNetwork:
    """
    Neural network architecture inspired by Leela Chess Zero / AlphaZero.

    Building blocks are Squeeze-and-Excitation (SE) residual blocks instead of
    plain ResNet blocks.  A typical configuration is 20 blocks with 256 filters.

    Input
    -----
    Shape: (8, 8, 20)  — channels-last, matching TensorFlow/Keras defaults.
    The 20 planes encode the current position only (no move history).

    Residual tower
    --------------
    One initial Conv → BN → ReLU projects the input to `num_filters` channels.
    Each SE block contains:
        Conv(3x3) → BN → ReLU
        Conv(3x3) → BN
        SE attention: GlobalAvgPool → Dense(C/se_ratio, relu)
                      → scale Dense(C, sigmoid) AND bias Dense(C, linear)
                      → (scale · conv_out) + bias   (Leela multiplicative + additive SE)
        Add(shortcut) → ReLU
    Body convs use no bias (a bias before BatchNorm is cancelled by BN's
    mean-subtraction; Leela's "convs have bias" describes the BN-folded
    inference graph).

    Policy head
    -----------
    Two Conv layers (no activation), Flatten and a final Dense producing
    `num_moves` logits (default 1858). Softmax activation on the final Dense.

    Value head  (WDL — Win / Draw / Loss)
    --------------------------------------
    Conv(1x1, FILTERS→32) → BN → ReLU → Conv(8x8 valid, →128) → ReLU
    → Flatten → Dense(3, softmax).
    The three outputs are the probabilities for winning, drawing, and losing.
    Loss: categorical cross-entropy against the actual game result one-hot vector.
    """

    def __init__(
        self,
        input_shape: tuple = (8, 8, 20),
        num_res_blocks: int = 20,
        num_filters: int = 256,
        num_moves: int = 1858,
        se_ratio: int = 8,
        learning_rate: float = 0.001,
    ):
        self.input_shape = input_shape
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.num_moves = num_moves
        self.se_ratio = se_ratio
        self.learning_rate = learning_rate

        self.model = self._build()
        self._compile()

        self._infer_fn = None          # cached tf.function (see _get_infer_fn)
        self._search_converter = None  # reused across search_for_best_move calls

    # ------------------------------------------------------------------
    # Block builders
    # ------------------------------------------------------------------

    def _build_se_block(self, x, block_name: str):
        """
        Squeeze-and-Excitation residual block (Leela-style, multiplicative + additive).

        Architecture:
            shortcut = x
            x = Conv(3x3) → BN → ReLU
            x = Conv(3x3) → BN
            # SE attention on x (before adding shortcut)
            se   = GlobalAvgPool(x)                 # (batch, C)
            se   = Dense(C // se_ratio, relu)(se)
            # Leela uses one FC → 2C split into a scale W and a bias B; here that
            # is two parallel Dense(C) layers (mathematically identical, and it
            # avoids slicing a single tensor).
            Z    = Dense(C, sigmoid)(se)            # per-channel gate in [0, 1]
            B    = Dense(C, linear)(se)             # per-channel additive bias
            x    = x * Z + B                        # (Z × input) + B
            x    = Add([shortcut, x])
            x    = ReLU(x)
        """
        C = self.num_filters
        bottleneck = max(1, C // self.se_ratio)
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

        # --- Excitation bottleneck ---
        se = layers.Dense(
            bottleneck,
            activation="relu",
            use_bias=True,
            name=f"{block_name}_se_fc1",
        )(se)

        # --- Scale (W → sigmoid) and additive bias (B → linear) ---
        scale = layers.Dense(
            C, activation="sigmoid", use_bias=True, name=f"{block_name}_se_scale_w"
        )(se)
        bias = layers.Dense(
            C, activation=None, use_bias=True, name=f"{block_name}_se_bias_b"
        )(se)

        # --- Reshape to (batch, 1, 1, C) for broadcasting ---
        scale = layers.Reshape((1, 1, C), name=f"{block_name}_se_scale_reshape")(scale)
        bias = layers.Reshape((1, 1, C), name=f"{block_name}_se_bias_reshape")(bias)

        # --- (Z × input) + B, then residual skip, then ReLU ---
        x = layers.Multiply(name=f"{block_name}_se_scale_mul")([x, scale])
        x = layers.Add(name=f"{block_name}_se_add")([x, bias, shortcut])
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

        Leela VALUE_WDL common part (Conv → Conv → Dense):
            Conv(1x1) FILTERS → 32          → BN → ReLU
            Conv(8x8 valid) 32 → 128         → ReLU   (collapses the 8x8 board
                                                       into a length-128 vector)
            Dense 128 → 3, softmax
        """
        # Channel-reduction conv (no bias: it is cancelled by the following BN)
        x = layers.Conv2D(32, 1, use_bias=False, name="value_conv1")(x)
        x = layers.BatchNormalization(name="value_bn")(x)
        x = layers.ReLU(name="value_relu1")(x)

        # 8x8 'valid' conv spans the whole board → (1, 1, 128); a convolution
        # producing the length-128 vector (not a Dense). Bias kept: no BN here.
        x = layers.Conv2D(128, 8, padding="valid", use_bias=True, name="value_conv2")(x)
        x = layers.ReLU(name="value_relu2")(x)

        x = layers.Flatten(name="value_flatten")(x)  # (1, 1, 128) → 128
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
            board_tensor: numpy array of shape (8, 8, 20)

        Returns:
            Tuple of (move_probabilities, wdl_probabilities)
                move_probabilities : numpy array of shape (num_moves,)
                wdl_probabilities  : numpy array of shape (3,)  — [win, draw, loss]
        """
        input_batch = np.expand_dims(board_tensor, axis=0).astype(np.float32)
        policy, value = self._get_infer_fn()(input_batch)
        return policy[0].numpy(), value[0].numpy()

    def predict_batch(self, board_tensors) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a batch of positions in a single call.

        Used by MCTS.search_batched so the GPU sees real batches.

        Args:
            board_tensors: array-like of shape (K, 8, 8, 20)

        Returns:
            Tuple of (policies (K, num_moves), wdl_values (K, 3)) as numpy arrays.
        """
        x = np.asarray(board_tensors, dtype=np.float32)
        policy, value = self._get_infer_fn()(x)
        return policy.numpy(), value.numpy()

    def _get_infer_fn(self):
        """Lazily build a tf.function for low-overhead inference.

        model.predict() carries heavy per-call overhead that dominates when it's
        invoked once per MCTS simulation (batch of 1); a traced tf.function is
        far cheaper. Built lazily so it traces after weights are loaded —
        load_weights mutates the existing variables in place, so a later load is
        reflected without rebuilding.
        """
        if self._infer_fn is None:
            @tf.function(input_signature=[
                tf.TensorSpec(shape=(None,) + tuple(self.input_shape),
                              dtype=tf.float32)
            ])
            def _infer(x):
                return self.model(x, training=False)
            self._infer_fn = _infer
        return self._infer_fn
    
    def search_for_best_move(
        self, board: chess.Board, num_simulations: int, batch_size: int = 16
    ) -> chess.Move:
        """
        Select the best move for the given position using MCTS search.

        Uses batched (virtual-loss) MCTS so leaf evaluations are sent to the
        network in groups of ``batch_size`` — far faster on GPU than batch-1.

        Args:
            board: The current chess board position
            num_simulations: Number of MCTS simulations to run
            batch_size: Leaves evaluated per network call (set 1 for the old
                sequential behaviour)

        Returns:
            The best move according to MCTS (greedy, most-visited)
        """
        if self._search_converter is None:
            self._search_converter = Converter()
        converter = self._search_converter
        mcts = MCTS(
            network=self,
            converter=converter,
            num_simulations=num_simulations,
        )
        root = Node(board)
        root = mcts.search_batched(root, add_noise=False, batch_size=batch_size)
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
            board_tensors  : numpy array of shape (batch_size, 8, 8, 20)
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