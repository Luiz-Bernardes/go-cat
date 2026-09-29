from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment


def test_agent_observes_environment():

    agent = Agent()
    environment = Environment()

    agent.observe(environment)

    assert len(agent.memory) == 1

    assert agent.memory.get()[0] == environment.get_perception()


def test_agent_observes_environment_over_time():

    agent = Agent()
    environment = Environment()

    agent.observe(environment)

    environment.step("down")
    agent.observe(environment)

    environment.step("down")
    agent.observe(environment)

    assert len(agent.memory) == 3