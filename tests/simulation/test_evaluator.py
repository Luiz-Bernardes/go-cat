import pytest
from unittest.mock import Mock, patch

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.simulation.runner import SimulationRunner
from ai_cat.simulation.evaluator import SimulationEvaluator


def make_metrics(
    steps,
    unique_positions,
    foods_found,
    cycles_detected,
    cycle_events,
):
    metrics = Mock()

    metrics.summary.return_value = {
        "steps": steps,
        "unique_positions": unique_positions,
        "foods_found": foods_found,
        "cycles_detected": cycles_detected,
        "cycle_events": cycle_events,
    }

    return metrics


def test_evaluator_with_real_agent():
    environment = Environment()
    agent = Agent()

    runner = SimulationRunner(environment, agent)
    evaluator = SimulationEvaluator(runner)

    agent.learning.exploration_rate = 0

    result = evaluator.evaluate(
        episodes=5,
        max_steps=20,
    )

    assert result["episodes"] == 5
    assert 0 <= result["successful_episodes"] <= 5
    assert 0 <= result["success_rate"] <= 1
    assert 0 < result["average_steps"] <= 20
    assert result["average_unique_positions"] >= 1
    assert result["average_cycles_detected"] >= 0
    assert result["average_cycle_events"] >= 0


def test_evaluator_does_not_train_agent():
    environment = Environment()
    agent = Agent()

    runner = SimulationRunner(environment, agent)
    evaluator = SimulationEvaluator(runner)

    with patch.object(agent, "learn") as learn_mock:
        evaluator.evaluate(
            episodes=5,
            max_steps=20,
        )

    learn_mock.assert_not_called()


def test_evaluator_calculates_statistics():
    runner = Mock()

    runner.run.side_effect = [
        make_metrics(10, 8, 1, 2, 1),
        make_metrics(20, 12, 0, 4, 2),
        make_metrics(30, 15, 1, 6, 3),
    ]

    evaluator = SimulationEvaluator(runner)

    result = evaluator.evaluate(
        episodes=3,
        max_steps=50,
    )

    assert result == {
        "episodes": 3,
        "successful_episodes": 2,
        "success_rate": 2 / 3,
        "average_steps": 20.0,
        "average_unique_positions": 35 / 3,
        "average_cycles_detected": 4.0,
        "average_cycle_events": 2.0,
    }


def test_evaluator_runs_requested_episodes():
    runner = Mock()

    runner.run.return_value = make_metrics(
        steps=5,
        unique_positions=4,
        foods_found=0,
        cycles_detected=0,
        cycle_events=0,
    )

    evaluator = SimulationEvaluator(runner)

    evaluator.evaluate(
        episodes=5,
        max_steps=20,
    )

    assert runner.run.call_count == 5
    runner.run.assert_called_with(max_steps=20)


@pytest.mark.parametrize("episodes", [0, -1])
def test_evaluator_rejects_invalid_episodes(episodes):
    runner = Mock()
    evaluator = SimulationEvaluator(runner)

    with pytest.raises(ValueError):
        evaluator.evaluate(episodes=episodes)


@pytest.mark.parametrize("max_steps", [0, -1])
def test_evaluator_rejects_invalid_max_steps(max_steps):
    runner = Mock()
    evaluator = SimulationEvaluator(runner)

    with pytest.raises(ValueError):
        evaluator.evaluate(max_steps=max_steps)
