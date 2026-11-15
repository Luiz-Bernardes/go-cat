import random

from ai_cat.simulation.q_table_analysis import (
    snapshot_q_table,
    compare_q_tables,
)

from ai_cat.simulation.q_table_analysis import (
    snapshot_q_table,
    compare_q_tables,
    compare_policies,
)

from ai_cat.simulation.trajectory_analysis import (
    record_trajectory,
    inspect_initial_decision,
    find_first_divergence,
)

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.training.runner import TrainingRunner
from ai_cat.simulation.runner import SimulationRunner
from ai_cat.simulation.evaluator import SimulationEvaluator
from ai_cat.simulation.trajectory_analysis import record_trajectory, find_best_actions

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

    # Captura o estado inicial do ambiente A
    initial_state = inspect_initial_decision(
        agent,
        environment_a,
    )["state"]

    before = evaluate_agent(
        agent,
        environment_a,
    )

    initial_decision_before = inspect_initial_decision(
        agent,
        environment_a,
    )

    trajectory_before = record_trajectory(
        agent,
        environment_a,
    )

    trainer_b = TrainingRunner(
        environment=environment_b,
        agent=agent,
    )

    agent.learning.start_tracking(
        state=initial_state,
        action="right",
    )

    trainer_b.run(
        episodes=1000,
        max_steps=100,
    )

    tracked_updates = agent.learning.tracked_updates.copy()

    next_state = tracked_updates[0]["next_state"]

    left_q_value = agent.learning.get_q_values(next_state)["left"]

    left_visits = agent.learning.get_visit_count(
        next_state,
        "left",
    )

    if verbose:
        print("\n=== LEFT ACTION INVESTIGATION ===")
        print(f"Next state: {next_state}")
        print(f"Final Q(left): {left_q_value:.12f}")
        print(f"Visits(left): {left_visits}")

    q_table_after = snapshot_q_table(agent)

    q_table_changes = compare_q_tables(
        q_table_before,
        q_table_after,
    )

    policy_changes = compare_policies(
        q_table_before,
        q_table_after,
    )

    after = evaluate_agent(
        agent,
        environment_a,
    )

    initial_decision_after = inspect_initial_decision(
        agent,
        environment_a,
    )

    trajectory_after = record_trajectory(
        agent,
        environment_a,
    )

    first_divergence = find_first_divergence(
        trajectory_before,
        trajectory_after,
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

        print("\n=== POLICY ANALYSIS ===")
        print(f"Shared states: {policy_changes['shared_states']}")
        print(f"Changed policies: {policy_changes['changed_policies']}")
        print(f"Unchanged policies: {policy_changes['unchanged_policies']}")

        print("\n=== TRAJECTORY BEFORE TRAINING B ===")

        for step in trajectory_before:
            print(
                f"Step {step['step']}: "
                f"Position {step['position']} | "
                f"Action {step['action']}"
            )

        print("\n=== TRAJECTORY AFTER TRAINING B ===")

        for step in trajectory_after:
            print(
                f"Step {step['step']}: "
                f"Position {step['position']} | "
                f"Action {step['action']}"
            )

        print("\n=== INITIAL DECISION ANALYSIS ===")

        print(
            "Same initial state:",
            initial_decision_before["state"]
            == initial_decision_after["state"],
        )

        print("\nQ-values:")

        for action in agent.actions:
            before_q = initial_decision_before["q_values"][action]
            after_q = initial_decision_after["q_values"][action]

            print(
                f"{action:>5} | "
                f"Before: {before_q:>8.4f} | "
                f"After: {after_q:>8.4f} | "
                f"Delta: {after_q - before_q:+.4f}"
            )

        print(
            "\nChosen action:",
            initial_decision_before["chosen_action"],
            "->",
            initial_decision_after["chosen_action"],
        )

        if first_divergence is not None:
            print(
                "First divergence at step:",
                first_divergence["step"],
            )

        print("\n=== INITIAL ACTION COMPARISON ===")

        valid_actions = ["down", "right"]

        for label, decision in [
            ("BEFORE", initial_decision_before),
            ("AFTER", initial_decision_after),
        ]:
            q_values = decision["q_values"]

            best_actions = find_best_actions(
                q_values,
                valid_actions,
            )

            print(f"\n{label} TRAINING B")

            for action in valid_actions:
                print(
                    f"{action}: {q_values[action]:.12f}"
                )

            print(f"Best actions (tolerance): {best_actions}")

            print(
                "Exact equality:",
                q_values["down"] == q_values["right"],
            )

            print(
                "Chosen action:",
                decision["chosen_action"],
            )

        print("\n=== Q-LEARNING UPDATE TRACKING ===")

        print(f"Total updates: {len(tracked_updates)}")

        if tracked_updates:
            print("\nFirst update:")
            print(tracked_updates[0])

            print("\nLast update:")
            print(tracked_updates[-1])

            print("\nQ-value evolution:")

            selected_updates = [1, 7, 8, 10, 20, 30, 40, 50]

            for index in selected_updates:
                if index > len(tracked_updates):
                    continue

                update = tracked_updates[index - 1]

                print(f"\n=== UPDATE {index} ===")

                print(f"Reward: {update['reward']}")
                print(f"Target: {update['target']:.8f}")
                print(f"TD Error: {update['td_error']:.8f}")

                print(
                    f"Q: {update['old_value']:.8f} -> "
                    f"{update['new_value']:.8f}"
                )

                print("Next state Q-values:")

                for action, value in update["next_q_values"].items():
                    print(f"  {action}: {value:.8f}")

                valid_actions = update["valid_next_actions"]
                next_q_values = update["next_q_values"]

                if valid_actions is None:
                    valid_actions = list(next_q_values.keys())

                best_next_q = (
                    max(next_q_values[action] for action in valid_actions)
                    if valid_actions
                    else 0.0
                )

                print("Valid next actions:", valid_actions)
                print("Best valid next Q:", best_next_q)

    return (
        before,
        after,
        q_table_changes,
        policy_changes,
        trajectory_before,
        trajectory_after,
        initial_decision_before,
        initial_decision_after,
        first_divergence,
    )


if __name__ == "__main__":
    run_experiment()
