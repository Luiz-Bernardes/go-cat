from unittest.mock import Mock, patch

from ai_cat.training.runner import TrainingRunner
from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment


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


def test_training_runner_learns_from_correct_transition():
    environment = Mock()
    agent = Mock()

    environment.cat_position = [0, 0]
    environment.step.return_value = (10, True)

    initial_state = ("initial_state",)
    next_state = ("next_state",)

    agent.get_state.side_effect = [
        initial_state,
        next_state,
    ]

    agent.choose_action.return_value = "right"

    runner = TrainingRunner(environment, agent)

    runner.run(episodes=1, max_steps=10)

    agent.choose_action.assert_called_once_with(
        initial_state
    )

    environment.step.assert_called_once_with(
        "right"
    )

    agent.learn.assert_called_once_with(
        initial_state,
        "right",
        10,
        next_state,
        True,
    )


def test_training_runner_resets_memory_but_preserves_learning():
    environment = Mock()
    agent = Mock()

    environment.cat_position = [0, 0]
    environment.step.return_value = (10, True)

    agent.get_state.return_value = ("state",)
    agent.choose_action.return_value = "right"

    runner = TrainingRunner(environment, agent)

    runner.run(episodes=3, max_steps=10)

    # O ambiente e a memória reiniciam a cada episódio.
    assert environment.reset.call_count == 3
    assert agent.reset_memory.call_count == 3

    # O aprendizado acontece nos três episódios.
    assert agent.learn.call_count == 3

    # A exploração diminui ao final de cada episódio.
    assert agent.decay_exploration.call_count == 3

    # O agente é observado no início e após cada movimento.
    assert agent.observe.call_count == 6


def test_training_runner_preserves_q_table_between_episodes():
    environment = Environment()
    agent = Agent()

    runner = TrainingRunner(environment, agent)

    # Conhecimento previamente adquirido.
    known_state = ("known_state",)

    agent.learning.q_table[known_state] = {
        "up": 0.0,
        "down": 0.0,
        "left": 0.0,
        "right": 25.0,
    }

    # Impede exploração aleatória.
    agent.learning.exploration_rate = 0

    with patch.object(environment, "step", return_value=(0, True)):
        runner.run(episodes=3, max_steps=10)

    # O conhecimento anterior continua armazenado.
    assert known_state in agent.learning.q_table

    assert agent.learning.q_table[known_state]["right"] == 25.0

    # A Q-table também recebeu estados do treinamento.
    assert len(agent.learning.q_table) > 1

    # A memória corresponde apenas ao último episódio.
    assert len(agent.memory) == 2
    assert len(agent.movement_history.get()) == 2
    assert len(agent.spatial_memory) == 1
