import random

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.training.runner import TrainingRunner
from ai_cat.simulation.generalization_experiment import evaluate_scenarios


def run_seeded_experiment(seed):
    random.seed(seed)

    agent = Agent()
    environment = Environment()

    agent.learning.exploration_rate = 1.0

    trainer = TrainingRunner(environment, agent)
    trainer.run(
        episodes=100,
        max_steps=100,
    )

    agent.learning.exploration_rate = 0

    return evaluate_scenarios(agent)


def test_generalization_experiment_is_reproducible():
    first = run_seeded_experiment(seed=42)
    second = run_seeded_experiment(seed=42)

    assert first == second
