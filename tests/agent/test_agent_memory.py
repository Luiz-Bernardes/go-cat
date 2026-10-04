from ai_cat.agents.agent import Agent
from ai_cat.perception.perception import Perception, PerceptionType
from ai_cat.environments.grid_world import Environment

def test_agent_has_memory():

    agent = Agent()

    assert agent.memory is not None
    assert len(agent.memory) == 0


def test_agent_has_spatial_memory():
    agent = Agent()

    assert agent.spatial_memory is not None


def test_agent_memory_stores_perceptions():

    agent = Agent()

    perception = Perception(
        up=PerceptionType.WALL,
        down=PerceptionType.EMPTY,
        left=PerceptionType.WALL,
        right=PerceptionType.EMPTY,
    )

    agent.memory.add(perception)

    assert len(agent.memory) == 1
    assert agent.memory.get() == (perception,)


def test_agent_observation_stores_spatial_position():
    agent = Agent()
    environment = Environment()

    agent.observe(environment)

    assert agent.spatial_memory.contains((0, 0))


def test_agent_spatial_memory_stores_multiple_positions():
    agent = Agent()
    environment = Environment()

    agent.observe(environment)

    environment.step("down")
    agent.observe(environment)

    environment.step("right")
    agent.observe(environment)

    assert agent.spatial_memory.contains((0, 0))
    assert agent.spatial_memory.contains((0, 1))
    assert agent.spatial_memory.contains((1, 1))

    assert len(agent.spatial_memory) == 3

def test_agent_reset_memory():
    agent = Agent()
    environment = Environment()

    agent.observe(environment)

    environment.step("down")
    agent.observe(environment)

    assert len(agent.memory) > 0
    assert len(agent.spatial_memory) > 0

    agent.reset_memory()

    assert len(agent.memory) == 0
    assert len(agent.spatial_memory) == 0