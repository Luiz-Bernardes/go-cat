import pytest

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.simulation.runner import SimulationRunner


def test_runner_records_initial_position():
    environment = Environment()
    agent = Agent()

    environment.foods = set()

    runner = SimulationRunner(environment, agent)

    metrics = runner.run(max_steps=1)

    assert metrics.unique_positions >= 1


def test_runner_respects_max_steps():
    environment = Environment()
    agent = Agent()

    runner = SimulationRunner(environment, agent)

    metrics = runner.run(max_steps=5)

    assert metrics.steps <= 5


def test_runner_records_food():
    environment = Environment()
    agent = Agent()

    runner = SimulationRunner(environment, agent)

    metrics = runner.run(max_steps=100)

    assert metrics.foods_found in (0, 1)


def test_runner_rejects_invalid_max_steps():
    environment = Environment()
    agent = Agent()

    runner = SimulationRunner(environment, agent)

    with pytest.raises(ValueError):
        runner.run(max_steps=0)
