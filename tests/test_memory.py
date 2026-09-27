import pytest

from ai_cat.memory.memory import Memory


def test_memory_starts_empty():
    memory = Memory(capacity=3)

    assert memory.get() == ()
    assert len(memory) == 0


def test_memory_stores_items():
    memory = Memory(capacity=3)

    memory.add("A")
    memory.add("B")

    assert memory.get() == ("A", "B")
    assert len(memory) == 2


def test_memory_respects_capacity():
    memory = Memory(capacity=3)

    memory.add("A")
    memory.add("B")
    memory.add("C")
    memory.add("D")

    assert memory.get() == ("B", "C", "D")
    assert len(memory) == 3


def test_memory_removes_oldest_item():
    memory = Memory(capacity=2)

    memory.add("A")
    memory.add("B")

    assert memory.get() == ("A", "B")

    memory.add("C")

    assert memory.get() == ("B", "C")


def test_memory_reset():
    memory = Memory(capacity=3)

    memory.add("A")
    memory.add("B")
    memory.add("C")

    memory.reset()

    assert memory.get() == ()
    assert len(memory) == 0


def test_memory_rejects_invalid_capacity():
    with pytest.raises(ValueError):
        Memory(capacity=0)

    with pytest.raises(ValueError):
        Memory(capacity=-1)