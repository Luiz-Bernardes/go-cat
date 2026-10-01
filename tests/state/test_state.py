from ai_cat.environments.grid_world import Environment
from ai_cat.learning.state import State
from ai_cat.memory.memory import Memory
from ai_cat.perception.perception import PerceptionType


def test_state_represents_incomplete_memory():

    environment = Environment()
    memory = Memory(capacity=3)

    perception = environment.get_perception()
    memory.add(perception)

    state = State(memory, capacity=3)

    result = state.build()

    expected_perception = (
        PerceptionType.WALL,
        PerceptionType.EMPTY,
        PerceptionType.WALL,
        PerceptionType.EMPTY,
    )

    expected = (
        State.UNKNOWN,
        State.UNKNOWN,
        expected_perception,
    )

    assert result == expected


def test_state_represents_complete_memory():

    environment = Environment()
    memory = Memory(capacity=3)

    for action in [None, "down", "down"]:

        if action is not None:
            environment.step(action)

        memory.add(
            environment.get_perception()
        )

    state = State(memory, capacity=3)

    result = state.build()

    assert len(result) == 3

    assert result == (
        (
            PerceptionType.WALL,
            PerceptionType.EMPTY,
            PerceptionType.WALL,
            PerceptionType.EMPTY,
        ),
        (
            PerceptionType.EMPTY,
            PerceptionType.EMPTY,
            PerceptionType.WALL,
            PerceptionType.EMPTY,
        ),
        (
            PerceptionType.EMPTY,
            PerceptionType.EMPTY,
            PerceptionType.WALL,
            PerceptionType.EMPTY,
        ),
    )


def test_state_is_hashable():

    environment = Environment()
    memory = Memory(capacity=3)

    memory.add(
        environment.get_perception()
    )

    state = State(memory, capacity=3)

    result = state.build()

    hash(result)