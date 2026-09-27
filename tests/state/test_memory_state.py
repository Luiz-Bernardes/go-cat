from ai_cat.environments.grid_world import Environment
from ai_cat.memory.memory import Memory

UNKNOWN = (
    "unknown",
    "unknown",
    "unknown",
    "unknown",
)

def perception_to_state(perception):
    return (
        perception["up"],
        perception["down"],
        perception["left"],
        perception["right"],
    )


def variable_length_state(memory):
    return tuple(
        perception_to_state(perception)
        for perception in memory.get()
    )


def fixed_length_state(memory, capacity):
    items = [
        perception_to_state(perception)
        for perception in memory.get()
    ]

    padding = [UNKNOWN] * (capacity - len(items))

    return tuple(padding + items)


def test_memory_state_representations():

    environment = Environment()
    memory = Memory(capacity=3)

    print()

    # P1
    perception = environment.get_perception()
    memory.add(perception)

    variable_state = variable_length_state(memory)
    fixed_state = fixed_length_state(memory, 3)

    print("Após P1:")
    print("Memória:", memory.get())
    print("Estado variável:", variable_state)
    print("Estado fixo:", fixed_state)

    assert len(variable_state) == 1
    assert len(fixed_state) == 3

    assert variable_state != fixed_state

    # P2
    environment.step("down")
    perception = environment.get_perception()
    memory.add(perception)

    variable_state = variable_length_state(memory)
    fixed_state = fixed_length_state(memory, 3)

    print()
    print("Após P2:")
    print("Estado variável:", variable_state)
    print("Estado fixo:", fixed_state)

    assert len(variable_state) == 2
    assert len(fixed_state) == 3

    assert variable_state != fixed_state

    # P3
    environment.step("down")
    perception = environment.get_perception()
    memory.add(perception)

    variable_state = variable_length_state(memory)
    fixed_state = fixed_length_state(memory, 3)

    print()
    print("Após P3:")
    print("Estado variável:", variable_state)
    print("Estado fixo:", fixed_state)

    assert len(variable_state) == 3
    assert len(fixed_state) == 3

    # Quando a memória está cheia, as duas representações coincidem.
    assert variable_state == fixed_state