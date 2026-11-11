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


def test_evaluate_scenarios_uses_custom_scenarios():
    agent = Agent()
    agent.learning.exploration_rate = 0

    custom_scenarios = {
        "custom_environment": {
            "foods": {(1, 0)},
            "obstacles": set(),
        },
    }

    results = evaluate_scenarios(
        agent,
        scenarios=custom_scenarios,
    )

    assert list(results.keys()) == ["custom_environment"]
    assert results["custom_environment"]["episodes"] == 100
    assert results["custom_environment"]["success_rate"] == 1.0
