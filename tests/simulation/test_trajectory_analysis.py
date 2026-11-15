import pytest

from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.simulation.trajectory_analysis import record_trajectory, find_first_divergence, inspect_initial_decision

def test_record_trajectory_starts_at_origin():
    agent = Agent()

    environment = Environment(
        foods={(1, 0)},
        obstacles=set(),
    )

    trajectory = record_trajectory(
        agent,
        environment,
    )

    assert trajectory[0] == {
        "step": 0,
        "position": (0, 0),
        "action": None,
    }


def test_record_trajectory_preserves_exploration_rate():
    agent = Agent()
    agent.learning.exploration_rate = 0.5

    environment = Environment(
        foods={(1, 0)},
        obstacles=set(),
    )

    record_trajectory(agent, environment)

    assert agent.learning.exploration_rate == 0.5


def test_record_trajectory_rejects_invalid_max_steps():
    agent = Agent()
    environment = Environment()

    with pytest.raises(ValueError):
        record_trajectory(
            agent,
            environment,
            max_steps=0,
        )


def test_find_first_divergence():
    before = [
        {"step": 0, "position": (0, 0), "action": None},
        {"step": 1, "position": (1, 0), "action": "right"},
        {"step": 2, "position": (1, 1), "action": "down"},
    ]

    after = [
        {"step": 0, "position": (0, 0), "action": None},
        {"step": 1, "position": (0, 1), "action": "down"},
        {"step": 2, "position": (1, 1), "action": "right"},
    ]

    result = find_first_divergence(before, after)

    assert result["step"] == 1
    assert result["before"]["action"] == "right"
    assert result["after"]["action"] == "down"


def test_identical_trajectories_have_no_divergence():
    trajectory = [
        {"step": 0, "position": (0, 0), "action": None},
        {"step": 1, "position": (1, 0), "action": "right"},
    ]

    assert find_first_divergence(trajectory, trajectory) is None


def test_inspect_initial_decision_returns_q_values():
    agent = Agent()

    environment = Environment(
        foods={(1, 0)},
        obstacles=set(),
    )

    result = inspect_initial_decision(
        agent,
        environment,
    )

    assert "state" in result
    assert "q_values" in result
    assert "chosen_action" in result

    assert set(result["q_values"]) == {
        "up",
        "down",
        "left",
        "right",
    }

    assert result["chosen_action"] in {
        "down",
        "right",
    }


def test_inspect_initial_decision_preserves_exploration():
    agent = Agent()
    agent.learning.exploration_rate = 0.5

    environment = Environment()

    inspect_initial_decision(
        agent,
        environment,
    )

    assert agent.learning.exploration_rate == 0.5
