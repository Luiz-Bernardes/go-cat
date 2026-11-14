from ai_cat.simulation.forgetting_experiment import run_experiment


SEEDS = [10, 20, 30, 40, 50]


def evaluate_seeds(seeds=SEEDS):
    results = []

    for seed in seeds:
        before, after, _ = run_experiment(
            seed=seed,
            verbose=False,
        )

        results.append({
            "seed": seed,
            "before_steps": before["average_steps"],
            "after_steps": after["average_steps"],
            "before_success": before["success_rate"],
            "after_success": after["success_rate"],
        })

    return results


def summarize(results):
    if not results:
        raise ValueError("Results cannot be empty")

    worse = 0
    better = 0
    unchanged = 0
    success_losses = 0

    for result in results:
        before = result["before_steps"]
        after = result["after_steps"]

        if after > before:
            worse += 1
        elif after < before:
            better += 1
        else:
            unchanged += 1

        if result["after_success"] < result["before_success"]:
            success_losses += 1

    mean_step_delta = sum(
        result["after_steps"] - result["before_steps"]
        for result in results
    ) / len(results)

    return {
        "worse": worse,
        "better": better,
        "unchanged": unchanged,
        "success_losses": success_losses,
        "mean_step_delta": mean_step_delta,
    }


def run_multi_seed_experiment():
    results = evaluate_seeds()
    summary = summarize(results)

    print("\n=== MULTI-SEED FORGETTING EXPERIMENT ===")

    for result in results:
        print(
            f"Seed {result['seed']} | "
            f"Before: {result['before_steps']:.2f} | "
            f"After: {result['after_steps']:.2f} | "
            f"Success: {result['before_success']:.0%}"
            f" -> {result['after_success']:.0%}"
        )

    print("\n=== SUMMARY ===")
    print(f"Worse: {summary['worse']}")
    print(f"Better: {summary['better']}")
    print(f"Unchanged: {summary['unchanged']}")
    print(f"Success losses: {summary['success_losses']}")
    print(f"Mean step delta: {summary['mean_step_delta']:+.2f}")

    return results


if __name__ == "__main__":
    run_multi_seed_experiment()
