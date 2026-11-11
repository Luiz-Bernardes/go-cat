from unittest.mock import patch

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.training.runner import TrainingRunner
from ai_cat.simulation.runner import SimulationRunner
from ai_cat.simulation.evaluator import SimulationEvaluator


SCENARIOS = {
    "original": {
        "foods": {(7, 7), (2, 6), (8, 2)},
        "obstacles": {(3, 1), (4, 1), (5, 1)},
    },
    "modified_food": {
        "foods": {(5, 5)},
        "obstacles": {(3, 1), (4, 1), (5, 1)},
    },
    "unknown_environment": {
        "foods": {(9, 9)},
        "obstacles": {(1, 0), (2, 0), (3, 0)},
    },
}


def evaluate_scenarios(agent, scenarios=None):
    if scenarios is None:
        scenarios = SCENARIOS

    results = {}

    for name, scenario in scenarios.items():
        environment = Environment(
            size=10,
            foods=scenario["foods"],
            obstacles=scenario["obstacles"],
        )

        runner = SimulationRunner(environment, agent)
        evaluator = SimulationEvaluator(runner)

        results[name] = evaluator.evaluate(
            episodes=100,
            max_steps=100,
        )

    return results


def print_results(title, results):
    print(f"\n=== {title} ===")

    for scenario, metrics in results.items():
        print(f"\nScenario: {scenario}")
        print(f"Success rate: {metrics['success_rate']:.2%}")
        print(f"Average steps: {metrics['average_steps']:.2f}")
        print(
            "Average unique positions: "
            f"{metrics['average_unique_positions']:.2f}"
        )
        print(
            "Average cycle events: "
            f"{metrics['average_cycle_events']:.2f}"
        )


def run_experiment():
    agent = Agent()

    # Avaliação inicial sem treinamento.
    agent.learning.exploration_rate = 0
    before = evaluate_scenarios(agent)

    # Treinamento somente no ambiente original.
    training_environment = Environment()
    trainer = TrainingRunner(training_environment, agent)

    agent.learning.exploration_rate = 1.0

    trainer.run(
        episodes=1000,
        max_steps=100,
    )

    # Avaliação após treinamento.
    agent.learning.exploration_rate = 0
    after = evaluate_scenarios(agent)

    print_results("BEFORE TRAINING", before)
    print_results("AFTER TRAINING", after)


if __name__ == "__main__":
    run_experiment()
