import random


class Agent:
    def __init__(self):
        self.position = [0, 0]

        self.actions = [
            "up",
            "down",
            "left",
            "right"
        ]

        self.q_table = {}

        self.learning_rate = 0.1
        self.discount_factor = 0.9

        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {
                action: 0.0
                for action in self.actions
            }

        return self.q_table[state]

    def choose_action(self, state):
        q_values = self.get_q_values(state)

        if random.random() < self.exploration_rate:
            return random.choice(self.actions)

        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        current_q = q_values[action]
        best_next_q = max(next_q_values.values())

        new_q = current_q + self.learning_rate * (
            reward
            + self.discount_factor * best_next_q
            - current_q
        )

        q_values[action] = new_q

    def decay_exploration(self):
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )