import keras
import numpy as np

from reinforcement_learning import mcts
from reinforcement_learning.reinf_learn import ReinfLearn

model = keras.models.load_model("../common/random_model.keras")
mcts_searcher = mcts.MCTS(model)
learner = ReinfLearn(model)
for i in range(0, 11):
    print("Training iteration: ", +str(i))
    all_pos = []
    all_move_probs = []
    all_values = []
    for j in range(0, 10):
        pos, move_probs, values = learner.play_game()
        all_pos += pos
        all_move_probs += move_probs
        all_values += values
    np_pos = np.array(all_pos)
    np_probs = np.array(all_move_probs)
    np_values = np.array(all_values)
    model.fit(np_pos, [np_probs, np_values], epochs=10, batch_size=32)
    if i % 10 == 0:
        model.save("reinforcement_learning/random_model_it" + str(i) + ".keras")
