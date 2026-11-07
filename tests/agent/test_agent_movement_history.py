import numpy as np

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment


def test_agent_records_movement_history():
    environment = Environment()
    agent = Agent()

    agent.observe(environment)

    environment.cat_position = np.array([1, 0])
    agent.observe(environment)

    environment.cat_position = np.array([0, 0])
    agent.observe(environment)

    assert agent.movement_history.get() == (
        (0, 0),
        (1, 0),
        (0, 0),
    )


def test_agent_detects_movement_cycle():
    environment = Environment()
    agent = Agent()

    positions = [
        (0, 0),
        (1, 0),
        (0, 0),
        (1, 0),
    ]

    for position in positions:
        environment.cat_position = np.array(position)
        agent.observe(environment)

    assert agent.movement_history.has_cycle(
        cycle_length=2
    )


def test_agent_resets_movement_history():
    environment = Environment()
    agent = Agent()

    agent.observe(environment)

    assert len(agent.movement_history.get()) == 1

    agent.reset_memory()

    assert agent.movement_history.get() == ()


def test_agent_escapes_movement_cycle():
    environment = Environment()
    agent = Agent()

    environment.foods = set()
    environment.obstacles = set()

    agent.learning.exploration_rate = 0

    positions = [
        (0, 0),
        (1, 0),
        (0, 0),
        (1, 0),
    ]

    for position in positions:
        environment.cat_position = np.array(position)
        agent.observe(environment)

    # Todas as alternativas válidas já foram visitadas.
    agent.spatial_memory.add((2, 0))
    agent.spatial_memory.add((1, 1))

    assert agent.movement_history.has_cycle(cycle_length=2)

    state = agent.get_state(environment)

    q_values = agent.get_q_values(state)
    q_values["left"] = 10.0
    q_values["down"] = 5.0
    q_values["right"] = 1.0

    action = agent.choose_action(state)

    assert action != "left"


def test_agent_escapes_cycle_during_exploration():
    environment = Environment()
    agent = Agent()

    environment.foods = set()
    environment.obstacles = set()

    # Força exploração totalmente aleatória.
    agent.learning.exploration_rate = 1.0

    # Simula um ciclo entre duas posições.
    positions = [
        (0, 0),
        (1, 0),
        (0, 0),
        (1, 0),
    ]

    for position in positions:
        environment.cat_position = np.array(position)
        agent.observe(environment)

    # Marca todas as direções válidas como visitadas.
    agent.spatial_memory.add((2, 0))
    agent.spatial_memory.add((1, 1))

    assert agent.movement_history.has_cycle(cycle_length=2)

    state = agent.get_state(environment)

    # Executa várias escolhas para verificar se a
    # ação que mantém o ciclo continua disponível.
    actions = {
        agent.choose_action(state)
        for _ in range(100)
    }

    assert "left" not in actions
    assert actions.issubset({"down", "right"})
    assert actions