from ai_cat.memory.spatial_memory import SpatialMemory


def test_spatial_memory_stores_positions():
    memory = SpatialMemory()

    memory.add((0, 0))
    memory.add((0, 1))

    assert memory.contains((0, 0))
    assert memory.contains((0, 1))


def test_spatial_memory_does_not_contain_unknown_position():
    memory = SpatialMemory()

    memory.add((0, 0))

    assert not memory.contains((1, 1))


def test_spatial_memory_does_not_duplicate_positions():
    memory = SpatialMemory()

    memory.add((0, 0))
    memory.add((0, 0))

    assert len(memory) == 1


def test_spatial_memory_get_returns_positions():
    memory = SpatialMemory()

    memory.add((0, 0))
    memory.add((0, 1))

    assert set(memory.get()) == {
        (0, 0),
        (0, 1),
    }


def test_spatial_memory_reset():
    memory = SpatialMemory()

    memory.add((0, 0))
    memory.add((0, 1))

    memory.reset()

    assert len(memory) == 0
    assert not memory.contains((0, 0))


def test_spatial_memory_observes_visited_directions():
    memory = SpatialMemory()

    memory.add((0, 0))
    memory.add((0, 1))

    observation = memory.observe_position((0, 1))

    assert observation["up"] is True
    assert observation["down"] is False
    assert observation["left"] is False
    assert observation["right"] is False