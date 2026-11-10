import pytest

from unittest.mock import patch
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

    with patch.object(environment, "reset"):
        environment.cat_position = [0, 0]
        environment.foods = set()
        environment.obstacles = set()

        with patch.object(agent, "choose_action", return_value="right"):
            metrics = runner.run(max_steps=5)

    assert metrics.steps == 5
    assert metrics.foods_found == 0
    assert metrics.unique_positions == 6


def test_runner_records_food():
    environment = Environment()
    agent = Agent()

    runner = SimulationRunner(environment, agent)

    # Impede que o reset recrie o cenário padrão.
    with patch.object(environment, "reset"):
        environment.cat_position = [0, 0]
        environment.foods = {(1, 0)}
        environment.obstacles = set()

        # Força o agente a caminhar para a direita.
        with patch.object(agent, "choose_action", return_value="right"):
            metrics = runner.run(max_steps=10)

    assert metrics.steps == 1
    assert metrics.foods_found == 1
    assert metrics.unique_positions == 2


def test_runner_rejects_invalid_max_steps():
    environment = Environment()
    agent = Agent()

    runner = SimulationRunner(environment, agent)

    with pytest.raises(ValueError):
        runner.run(max_steps=0)


def test_runner_records_movement_cycle():
    environment = Environment()
    agent = Agent()

    runner = SimulationRunner(environment, agent)

    # Força movimentos alternados entre duas posições.
    actions = [
        "right",
        "left",
        "right",
        "left",
    ]

    with patch.object(environment, "reset"):
        environment.cat_position = [0, 0]
        environment.foods = set()
        environment.obstacles = set()

        with patch.object(agent, "choose_action", side_effect=actions):
            metrics = runner.run(max_steps=4)

    assert metrics.steps == 4
    assert metrics.unique_positions == 2
    assert metrics.foods_found == 0
    assert metrics.cycles_detected == 2
