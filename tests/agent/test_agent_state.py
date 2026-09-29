from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment


def test_agent_builds_state_from_memory():

    agent = Agent()
    environment = Environment()

    agent.observe(environment)

    state = agent.get_state()

    assert len(state) == 3

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
            "wall",
            "empty",
            "wall",
            "empty",
        ),
    )


def test_agent_state_changes_as_memory_grows():

    agent = Agent()
    environment = Environment()

    agent.observe(environment)
    state_1 = agent.get_state()

    environment.step("down")
    agent.observe(environment)
    state_2 = agent.get_state()

    environment.step("down")
    agent.observe(environment)
    state_3 = agent.get_state()

    assert state_1 != state_2
    assert state_2 != state_3

    assert len(state_1) == 3
    assert len(state_2) == 3
    assert len(state_3) == 3