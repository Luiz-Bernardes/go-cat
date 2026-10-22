import pytest

from ai_cat.memory.movement_history import MovementHistory

def test_stores_positions_in_order():
    history = MovementHistory()

    history.add((0, 0))
    history.add((0, 1))
    history.add((1, 1))

    assert history.get() == (
        (0, 0),
        (0, 1),
        (1, 1),
    )


def test_detects_two_position_cycle():
    history = MovementHistory()

    for position in [
        (0, 0),
        (0, 1),
        (0, 0),
        (0, 1),
    ]:
        history.add(position)

    assert history.has_cycle(cycle_length=2)


def test_detects_three_position_cycle():
    history = MovementHistory()

    for position in [
        (0, 0),
        (0, 1),
        (1, 1),
        (0, 0),
        (0, 1),
        (1, 1),
    ]:
        history.add(position)

    assert history.has_cycle(cycle_length=3)


def test_does_not_detect_incomplete_cycle():
    history = MovementHistory()

    history.add((0, 0))
    history.add((0, 1))
    history.add((0, 0))

    assert not history.has_cycle(cycle_length=2)


def test_history_respects_capacity():
    history = MovementHistory(capacity=3)

    history.add((0, 0))
    history.add((0, 1))
    history.add((1, 1))
    history.add((1, 2))

    assert history.get() == (
        (0, 1),
        (1, 1),
        (1, 2),
    )


def test_reset_clears_history():
    history = MovementHistory()
    history.add((0, 0))
    history.add((0, 1))

    history.reset()

    assert history.get() == ()
    assert not history.has_cycle()


@pytest.mark.parametrize("capacity", [0, -1])
def test_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        MovementHistory(capacity=capacity)


@pytest.mark.parametrize("cycle_length", [0, -1])
def test_rejects_invalid_cycle_length(cycle_length):
    history = MovementHistory()

    with pytest.raises(ValueError):
        history.has_cycle(cycle_length=cycle_length)
