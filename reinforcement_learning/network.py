from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf


class Network:
    def __init__(self):
        pass

    def build_and_save_model(
        self, filepath="reinforcement_learning/random_model.keras"
    ):
        inp = Input((21,))

        l1 = Dense(128, activation=tf.nn.relu)(inp)
        l2 = Dense(128, activation=tf.nn.relu)(l1)
        l3 = Dense(128, activation=tf.nn.relu)(l2)
        l4 = Dense(128, activation=tf.nn.relu)(l3)
        l5 = Dense(128, activation=tf.nn.relu)(l4)
        policyOut = Dense(28, name="policyHead", activation="softmax")(l5)
        valueOut = Dense(1, activation=tf.nn.tanh, name="valueHead")(l5)
        bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.model = Model(inp, [policyOut, valueOut])
        self.model.compile(
            optimizer="SGD", loss={"valueHead": "mean_squared_error", "policyHead": bce}
        )
        self.model.save(filepath)

    def load_model(self, filepath="reinforcement_learning/random_model.keras"):
        self.model = tf.keras.models.load_model(filepath)
