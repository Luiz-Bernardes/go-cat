import random
from ai_cat.learning.q_learning import QLearning
from ai_cat.memory.memory import Memory
from ai_cat.memory.spatial_memory import SpatialMemory
from ai_cat.learning.state import State
from ai_cat.perception.perception import PerceptionType
from ai_cat.memory.movement_history import MovementHistory

class Agent:
    def __init__(self):
        self.memory = Memory(capacity=3)
        self.spatial_memory = SpatialMemory()
        self.movement_history = MovementHistory(capacity=12)

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

        spatial_observation = state[1]

        visited_directions = {
            "up": spatial_observation[0],
            "down": spatial_observation[1],
            "left": spatial_observation[2],
            "right": spatial_observation[3],
        }

        unvisited_actions = [
            action
            for action in valid_actions
            if not visited_directions[action]
        ]

        if unvisited_actions:
            valid_actions = unvisited_actions


        if self.movement_history.get_cycle_length() is not None:
            positions = self.movement_history.get()

            if len(positions) >= 2:
                current_position = positions[-1]
                cycle_length = self.movement_history.get_cycle_length()
                next_position = positions[-cycle_length]

                dx = next_position[0] - current_position[0]
                dy = next_position[1] - current_position[1]

                reverse_action = None

                if dx == -1 and dy == 0:
                    reverse_action = "left"
                elif dx == 1 and dy == 0:
                    reverse_action = "right"
                elif dx == 0 and dy == -1:
                    reverse_action = "up"
                elif dx == 0 and dy == 1:
                    reverse_action = "down"

                alternatives = [
                    action
                    for action in valid_actions
                    if action != reverse_action
                ]

                if alternatives:
                    valid_actions = alternatives

        if random.random() < self.learning.exploration_rate:
            return random.choice(valid_actions)

        return max(
            valid_actions,
            key=lambda action: q_values[action],
        )

    def learn(self, state, action, reward, next_state, done=False):
        self.learning.learn(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def decay_exploration(self):
        self.learning.decay_exploration()

    def observe(self, environment):
        perception = environment.get_perception()
        position = tuple(environment.cat_position)

        self.memory.add(perception)
        self.spatial_memory.add(position)
        self.movement_history.add(position)

    def reset_memory(self):
        self.memory.reset()
        self.spatial_memory.reset()
        self.movement_history.reset()

    def observe_spatially(self, environment):
        return self.spatial_memory.observe_position(
            environment.cat_position
        )