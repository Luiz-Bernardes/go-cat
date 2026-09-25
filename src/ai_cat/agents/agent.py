from ai_cat.learning.q_learning import QLearning

class Agent:
    def __init__(self):
        self.position = [0, 0]

        self.actions = [
            "up",
            "down",
            "left",
            "right",
        ]

        self.learning = QLearning(self.actions)

    def get_q_values(self, state):
        return self.learning.get_q_values(state)

    def choose_action(self, state):
        return self.learning.choose_action(state)

    def learn(self, state, action, reward, next_state):
        self.learning.learn(
            state,
            action,
            reward,
            next_state,
        )

    def decay_exploration(self):
        self.learning.decay_exploration()