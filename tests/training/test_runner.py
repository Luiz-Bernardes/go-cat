from unittest.mock import Mock

from ai_cat.training.runner import TrainingRunner


def test_training_runner_updates_agent():
    environment = Mock()
    agent = Mock()

    environment.cat_position = [0, 0]
    environment.step.return_value = (-1, False)

    agent.get_state.return_value = ("state",)
    agent.choose_action.return_value = "right"

    runner = TrainingRunner(environment, agent)

    runner.run(episodes=1, max_steps=3)

    assert environment.reset.call_count == 1
    assert agent.reset_memory.call_count == 1
    assert environment.step.call_count == 3
    assert agent.learn.call_count == 3


def test_training_runner_stops_when_food_is_found():
    environment = Mock()
    agent = Mock()

    environment.cat_position = [0, 0]

    environment.step.side_effect = [
        (-1, False),
        (10, True),
    ]

    agent.get_state.return_value = ("state",)
    agent.choose_action.return_value = "right"

    runner = TrainingRunner(environment, agent)

    runner.run(episodes=1, max_steps=10)

    assert environment.step.call_count == 2
    assert agent.learn.call_count == 2
    assert agent.decay_exploration.call_count == 1