import random

from ai_cat.simulation.q_table_analysis import (
    snapshot_q_table,
    compare_q_tables,
)

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.training.runner import TrainingRunner
from ai_cat.simulation.runner import SimulationRunner
from ai_cat.simulation.evaluator import SimulationEvaluator


def evaluate_agent(agent, environment):
    previous_exploration = agent.learning.exploration_rate

    agent.learning.exploration_rate = 0

    runner = SimulationRunner(environment, agent)
    evaluator = SimulationEvaluator(runner)

    results = evaluator.evaluate(
        episodes=100,
        max_steps=100,
    )

    agent.learning.exploration_rate = previous_exploration

    return results


def run_experiment(seed=42, verbose=True):
    random.seed(seed)

    agent = Agent()

    environment_a = Environment(
        foods={(7, 7), (2, 6), (8, 2)},
        obstacles={(3, 1), (4, 1), (5, 1)},
    )

    environment_b = Environment(
        foods={(5, 5)},
        obstacles={(3, 1), (4, 1), (5, 1)},
    )

    trainer_a = TrainingRunner(
        environment=environment_a,
        agent=agent,
    )

    trainer_a.run(
        episodes=1000,
        max_steps=100,
    )

    q_table_before = snapshot_q_table(agent)

    before = evaluate_agent(
        agent,
        environment_a,
    )

    trainer_b = TrainingRunner(
        environment=environment_b,
        agent=agent,
    )

    trainer_b.run(
        episodes=1000,
        max_steps=100,
    )

    q_table_after = snapshot_q_table(agent)

    q_table_changes = compare_q_tables(
        q_table_before,
        q_table_after,
    )

    after = evaluate_agent(
        agent,
        environment_a,
    )

    if verbose:
        print("\n=== FORGETTING EXPERIMENT ===")

        print("\nBEFORE TRAINING ON B")
        print(f"Success: {before['success_rate']:.2%}")
        print(f"Steps: {before['average_steps']:.2f}")

        print("\nAFTER TRAINING ON B")
        print(f"Success: {after['success_rate']:.2%}")
        print(f"Steps: {after['average_steps']:.2f}")

        print("\n=== Q-TABLE ANALYSIS ===")
        print(f"States before: {q_table_changes['states_before']}")
        print(f"States after: {q_table_changes['states_after']}")
        print(f"Shared states: {q_table_changes['shared_states']}")
        print(f"Changed states: {q_table_changes['changed_states']}")
        print(f"Changed actions: {q_table_changes['changed_actions']}")

    return before, after, q_table_changes


if __name__ == "__main__":
    run_experiment()
