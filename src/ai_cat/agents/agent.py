from ai_cat.learning.q_learning import QLearning
from ai_cat.memory.memory import Memory
from ai_cat.memory.spatial_memory import SpatialMemory
from ai_cat.learning.state import State

class Agent:
    def __init__(self):
        self.memory = Memory(capacity=3)
        self.spatial_memory = SpatialMemory()

        self.state = State(
            self.memory,
            capacity=3,
        )

        self.actions = [
            "up",
            "down",
            "left",
            "right",
        ]

        self.learning = QLearning(self.actions)

    def get_q_values(self, state):
        return self.learning.get_q_values(state)

    def get_state(self):
        return self.state.build()

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

    def observe(self, environment):
        perception = environment.get_perception()

        self.memory.add(perception)
        self.spatial_memory.add(environment.cat_position)

    def reset_memory(self):
        self.memory.reset()
        self.spatial_memory.reset()