from ai_cat.agents.agent import Agent
from ai_cat.perception.perception import Perception, PerceptionType


def test_agent_has_memory():

    agent = Agent()

    assert agent.memory is not None
    assert len(agent.memory) == 0


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