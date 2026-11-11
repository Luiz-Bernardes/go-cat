import random

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.training.runner import TrainingRunner
from ai_cat.simulation.generalization_experiment import (
    evaluate_scenarios,
    print_results,
)

UNSEEN_SCENARIOS = {
    "unseen_1": {
        "foods": {(6, 3)},
        "obstacles": {(2, 1), (3, 1), (4, 1)},
    },
    "unseen_2": {
        "foods": {(1, 8)},
        "obstacles": {(3, 3), (3, 4), (3, 5)},
    },
    "unseen_3": {
        "foods": {(8, 8)},
        "obstacles": {(5, 5), (5, 6), (6, 5)},
    },
    "unseen_4": {
        "foods": {(4, 7)},
        "obstacles": {(1, 3), (2, 3), (3, 3)},
    },
    "unseen_5": {
        "foods": {(9, 2)},
        "obstacles": {(6, 1), (6, 2), (6, 3)},
    },
}

def train_fixed_agent(seed=42):
    random.seed(seed)

    agent = Agent()
    environment = Environment()

    trainer = TrainingRunner(environment, agent)

    trainer.run(
        episodes=1000,
        max_steps=100,
    )

    agent.learning.exploration_rate = 0

    return agent


def train_varied_agent(seed=42):
    random.seed(seed)

    agent = Agent()

    environments = [
        Environment(),
        Environment(
            foods={(5, 5)},
            obstacles={(3, 1), (4, 1), (5, 1)},
        ),
        Environment(
            foods={(2, 8), (8, 4)},
            obstacles={(1, 2), (2, 2), (3, 2)},
        ),
    ]

    trainer = TrainingRunner(
        environments=environments,
        agent=agent,
    )

    trainer.run(
        episodes=1000,
        max_steps=100,
    )

    agent.learning.exploration_rate = 0

    return agent


def run_comparison():
    fixed_agent = train_fixed_agent()
    varied_agent = train_varied_agent()

    fixed_results = evaluate_scenarios(fixed_agent)
    varied_results = evaluate_scenarios(varied_agent)

    fixed_unseen = evaluate_scenarios(
        fixed_agent,
        UNSEEN_SCENARIOS,
    )

    varied_unseen = evaluate_scenarios(
        varied_agent,
        UNSEEN_SCENARIOS,
    )

    print_results(
        "FIXED TRAINING - UNSEEN",
        fixed_unseen,
    )

    print_results(
        "VARIED TRAINING - UNSEEN",
        varied_unseen,
    )


if __name__ == "__main__":
    run_comparison()
