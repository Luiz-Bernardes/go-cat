from unittest.mock import patch

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.training.runner import TrainingRunner
from ai_cat.simulation.runner import SimulationRunner
from ai_cat.simulation.evaluator import SimulationEvaluator


def test_trained_agent_in_modified_environment():
    environment = Environment()
    agent = Agent()

    training_runner = TrainingRunner(environment, agent)

    # Treina no ambiente original.
    training_runner.run(
        episodes=1000,
        max_steps=100,
    )

    # Ambiente de avaliação com comida em outra posição.
    evaluation_environment = Environment()
    evaluation_environment.foods = {(5, 5)}

    simulation_runner = SimulationRunner(
        evaluation_environment,
        agent,
    )

    evaluator = SimulationEvaluator(simulation_runner)

    agent.learning.exploration_rate = 0

    # Preserva a configuração modificada durante a avaliação.
    with patch.object(evaluation_environment, "reset"):
        results = evaluator.evaluate(
            episodes=10,
            max_steps=100,
        )

    assert results["episodes"] == 10
    assert 0 <= results["success_rate"] <= 1
    assert 0 < results["average_steps"] <= 100
