import random
from ai_cat.learning.q_learning import QLearning
from ai_cat.memory.memory import Memory
from ai_cat.memory.spatial_memory import SpatialMemory
from ai_cat.learning.state import State
from ai_cat.perception.perception import PerceptionType

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

    def get_state(self, environment=None):
        if environment is None:
            return self.state.build()

        spatial_observation = self.observe_spatially(environment)

        return self.state.build(
            spatial_observation=spatial_observation
        )

    def choose_action(self, state):
        if not self.memory.get():
            return self.learning.choose_action(state)

        perception = self.memory.get()[-1]

        directions = {
            "up": perception.up,
            "down": perception.down,
            "left": perception.left,
            "right": perception.right,
        }

        valid_actions = [
            action
            for action, perception_type in directions.items()
            if perception_type not in (
                PerceptionType.WALL,
                PerceptionType.OBSTACLE,
            )
        ]

        if not valid_actions:
            return self.learning.choose_action(state)

        q_values = self.learning.get_q_values(state)

        if (
            random.random()
            < self.learning.exploration_rate
        ):
            return random.choice(valid_actions)

        return max(
            valid_actions,
            key=lambda action: q_values[action],
        )

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

    def observe_spatially(self, environment):
        return self.spatial_memory.observe_position(
            environment.cat_position
        )