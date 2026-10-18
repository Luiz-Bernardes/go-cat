from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.perception.perception import PerceptionType

def test_detects_repeated_positions():
    path = [
        (1, 1),
        (2, 1),
        (1, 1),
        (2, 1),
        (1, 1),
    ]

    visited_positions = set()
    repeated_positions = []

    for position in path:
        if position in visited_positions:
            repeated_positions.append(position)
        else:
            visited_positions.add(position)

    assert len(repeated_positions) == 3
    assert len(visited_positions) == 2


def test_agent_can_repeat_positions():
    environment = Environment()
    agent = Agent()

    agent.learning.exploration_rate = 0

    environment.foods = set()
    environment.obstacles = set()

    agent.reset_memory()
    agent.observe(environment)

    visited_positions = set()
    repeated_positions = []

    for _ in range(20):
        state = agent.get_state(environment)
        action = agent.choose_action(state)

        environment.step(action)

        position = tuple(environment.cat_position)

        if position in visited_positions:
            repeated_positions.append(position)
        else:
            visited_positions.add(position)

        agent.observe(environment)

    assert len(repeated_positions) > 0


def test_detects_movement_cycle():
    path = [
        (1, 1),
        (2, 1),
        (1, 1),
        (2, 1),
        (1, 1),
        (2, 1),
    ]

    cycle_length = 2
    has_cycle = (
        len(path) >= cycle_length * 2
        and path[-cycle_length:] == path[-cycle_length * 2:-cycle_length]
    )

    assert has_cycle is True


def test_agent_avoids_wall_when_choosing_action():
    environment = Environment()
    agent = Agent()

    agent.learning.exploration_rate = 0

    environment.foods = set()
    environment.obstacles = set()

    agent.reset_memory()
    agent.observe(environment)

    state = agent.get_state(environment)
    action = agent.choose_action(state)

    perception = environment.get_perception()

    directions = {
        "up": perception.up,
        "down": perception.down,
        "left": perception.left,
        "right": perception.right,
    }

    assert directions[action] not in (
        PerceptionType.WALL,
        PerceptionType.OBSTACLE,
    )


def test_agent_avoids_recently_visited_position():
    environment = Environment()
    agent = Agent()

    environment.foods = set()
    environment.obstacles = set()

    # Simula um histórico de posições visitadas.
    agent.spatial_memory.add((0, 0))
    agent.spatial_memory.add((0, 1))
    agent.spatial_memory.add((1, 1))

    # O agente está em (1, 0).
    # A posição (0, 0) já foi visitada e fica à esquerda.
    environment.cat_position = __import__("numpy").array([1, 0])

    agent.reset_memory()
    agent.spatial_memory.add((0, 0))
    agent.spatial_memory.add((0, 1))
    agent.spatial_memory.add((1, 1))
    agent.spatial_memory.add((1, 0))

    agent.observe(environment)

    agent.learning.exploration_rate = 0

    state = agent.get_state(environment)
    action = agent.choose_action(state)

    assert action != "left"
