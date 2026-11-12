from ai_cat.simulation.training_comparison import (
    train_fixed_agent,
    train_varied_agent,
    UNSEEN_SCENARIOS,
)
from ai_cat.simulation.generalization_experiment import (
    evaluate_scenarios,
)


SEEDS = [10, 20, 30, 40, 50]
TRAINING_DURATIONS = [1000, 3000, 6000]


def evaluate_training_duration(train_function, episodes):
    results = []

    for seed in SEEDS:
        agent = train_function(
            seed=seed,
            episodes=episodes,
        )

        scenarios = evaluate_scenarios(
            agent,
            UNSEEN_SCENARIOS,
        )

        success_rate = sum(
            result["success_rate"]
            for result in scenarios.values()
        ) / len(scenarios)

        average_steps = sum(
            result["average_steps"]
            for result in scenarios.values()
        ) / len(scenarios)

        results.append({
            "seed": seed,
            "success_rate": success_rate,
            "average_steps": average_steps,
        })

    return results


def summarize(results):
    return {
        "success_rate": sum(
            result["success_rate"]
            for result in results
        ) / len(results),
        "average_steps": sum(
            result["average_steps"]
            for result in results
        ) / len(results),
    }


def run_experiment():
    for episodes in TRAINING_DURATIONS:
        print(f"\n=== TRAINING: {episodes} EPISODES ===")

        fixed_results = evaluate_training_duration(
            train_fixed_agent,
            episodes,
        )

        varied_results = evaluate_training_duration(
            train_varied_agent,
            episodes,
        )

        fixed_summary = summarize(fixed_results)
        varied_summary = summarize(varied_results)

        print(
            f"FIXED  | "
            f"Success: {fixed_summary['success_rate']:.2%} | "
            f"Steps: {fixed_summary['average_steps']:.2f}"
        )

        print(
            f"VARIED | "
            f"Success: {varied_summary['success_rate']:.2%} | "
            f"Steps: {varied_summary['average_steps']:.2f}"
        )


if __name__ == "__main__":
    run_experiment()
