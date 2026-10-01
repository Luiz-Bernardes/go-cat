from ai_cat.environments.grid_world import Environment
from ai_cat.learning.state import State
from ai_cat.memory.memory import Memory


def variable_length_state(memory):
    return tuple(
        State.perception_to_state(perception)
        for perception in memory.get()
    )


def test_memory_state_representations():

    environment = Environment()
    memory = Memory(capacity=3)
    state_builder = State(memory, capacity=3)

    # P1
    perception = environment.get_perception()
    memory.add(perception)

    variable_state = variable_length_state(memory)
    fixed_state = state_builder.build()

    assert len(variable_state) == 1
    assert len(fixed_state) == 3

    assert variable_state != fixed_state

    # P2
    environment.step("down")
    perception = environment.get_perception()
    memory.add(perception)

    variable_state = variable_length_state(memory)
    fixed_state = state_builder.build()

    assert len(variable_state) == 2
    assert len(fixed_state) == 3

    assert variable_state != fixed_state

    # P3
    environment.step("down")
    perception = environment.get_perception()
    memory.add(perception)

    variable_state = variable_length_state(memory)
    fixed_state = state_builder.build()

    assert len(variable_state) == 3
    assert len(fixed_state) == 3

    assert variable_state == fixed_state