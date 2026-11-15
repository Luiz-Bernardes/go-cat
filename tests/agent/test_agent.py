from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.perception.perception import PerceptionType

def test_agent_creates_q_values_for_new_state():
    agent = Agent()

    state = (0, 0)

    q_values = agent.get_q_values(state)

    assert state in agent.learning.q_table
    assert q_values == {
        "up": 0.0,
        "down": 0.0,
        "left": 0.0,
        "right": 0.0,
    }

def test_agent_starts_with_empty_q_table():
    agent = Agent()

    assert agent.learning.q_table == {}


def test_agent_gets_state_from_memory():
    environment = Environment()
    agent = Agent()

    environment.cat_position = [2, 1]

    agent.observe(environment)

    state = agent.get_state()

    assert state == (
        (
            "unknown",
            "unknown",
            "unknown",
            "unknown",
        ),
        (
            "unknown",
            "unknown",
            "unknown",
            "unknown",
        ),
        (
            PerceptionType.EMPTY,
            PerceptionType.EMPTY,
            PerceptionType.EMPTY,
            PerceptionType.OBSTACLE,
        ),
    )


def test_choose_action_prefers_first_action_on_q_value_tie():
    agent = Agent()

    environment = Environment(
        foods={(5, 5)},
        obstacles=set(),
    )

    environment.reset()
    agent.reset_memory()
    agent.observe(environment)

    state = agent.get_state(environment)

    agent.learning.exploration_rate = 0

    agent.learning.q_table[state] = {
        "up": 0.0,
        "down": -1.0,
        "left": 0.0,
        "right": -1.0,
    }

    action = agent.choose_action(state)

    assert action == "down"


def test_agent_learn_passes_valid_next_actions():
    agent = Agent()

    state = ("current_state",)
    next_state = ("next_state",)

    agent.learning.q_table[next_state] = {
        "up": 0.0,
        "down": -0.5,
        "left": -0.8,
        "right": -0.9,
    }

    agent.learn(
        state=state,
        action="right",
        reward=-1,
        next_state=next_state,
        done=False,
        valid_next_actions=["down", "left", "right"],
    )

    expected_target = (
        -1 + agent.learning.discount_factor * -0.5
    )

    expected_q = (
        agent.learning.learning_rate * expected_target
    )

    assert agent.learning.q_table[state]["right"] == expected_q
