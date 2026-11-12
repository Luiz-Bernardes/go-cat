from ai_cat.simulation.training_comparison import (
    train_fixed_agent,
    train_varied_agent,
    UNSEEN_SCENARIOS,
)
from ai_cat.simulation.generalization_experiment import evaluate_scenarios


SEEDS = [10, 20, 30, 40, 50]


def evaluate_strategy(train_function):
    results = []

    for seed in SEEDS:
        agent = train_function(seed=seed)

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
        "average_success_rate": sum(
            result["success_rate"]
            for result in results
        ) / len(results),
        "average_steps": sum(
            result["average_steps"]
            for result in results
        ) / len(results),
    }


def print_results(title, results):
    print(f"\n=== {title} ===")

    for result in results:
        print(
            f"Seed {result['seed']}: "
            f"Success {result['success_rate']:.2%}, "
            f"Steps {result['average_steps']:.2f}"
        )

    summary = summarize(results)

    print("\nAVERAGE")
    print(
        f"Success rate: "
        f"{summary['average_success_rate']:.2%}"
    )
    print(
        f"Average steps: "
        f"{summary['average_steps']:.2f}"
    )


def run_experiment():
    fixed_results = evaluate_strategy(
        train_fixed_agent
    )

    varied_results = evaluate_strategy(
        train_varied_agent
    )

    print_results(
        "FIXED TRAINING",
        fixed_results,
    )

    print_results(
        "VARIED TRAINING",
        varied_results,
    )


if __name__ == "__main__":
    run_experiment()
