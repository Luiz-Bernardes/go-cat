from ai_cat.simulation.forgetting_experiment import (
    evaluate_agent,
)
from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment


def test_evaluation_preserves_exploration_rate():
    agent = Agent()

    environment = Environment(
        foods={(1, 0)},
        obstacles=set(),
    )

    agent.learning.exploration_rate = 0.5

    evaluate_agent(agent, environment)

    assert agent.learning.exploration_rate == 0.5


def test_forgetting_experiment_can_run_silently(capsys):
    from ai_cat.simulation.forgetting_experiment import run_experiment

    before, after, changes = run_experiment(
        seed=42,
        verbose=False,
    )

    captured = capsys.readouterr()

    assert captured.out == ""

    assert before["success_rate"] == 1.0
    assert after["success_rate"] == 1.0

    assert before["average_steps"] > 0
    assert after["average_steps"] > 0

    assert changes["states_before"] > 0
    assert changes["states_after"] > 0
    assert changes["shared_states"] > 0

