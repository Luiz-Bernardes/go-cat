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
