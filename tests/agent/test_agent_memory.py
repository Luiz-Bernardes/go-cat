from ai_cat.agents.agent import Agent


def test_agent_has_memory():

    agent = Agent()

    assert agent.memory is not None
    assert len(agent.memory) == 0


def test_agent_memory_stores_perceptions():

    agent = Agent()

    perception = {
        "up": "wall",
        "down": "empty",
        "left": "wall",
        "right": "empty",
    }

    agent.memory.add(perception)

    assert len(agent.memory) == 1
    assert agent.memory.get() == (perception,)