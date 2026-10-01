from dataclasses import FrozenInstanceError

import pytest

from ai_cat.perception.perception import Perception, PerceptionType


def test_perception_stores_directions():

    perception = Perception(
        up=PerceptionType.WALL,
        down=PerceptionType.EMPTY,
        left=PerceptionType.WALL,
        right=PerceptionType.FOOD,
    )

    assert perception.up is PerceptionType.WALL
    assert perception.down is PerceptionType.EMPTY
    assert perception.left is PerceptionType.WALL
    assert perception.right is PerceptionType.FOOD


def test_perception_can_be_converted_to_tuple():

    perception = Perception(
        up=PerceptionType.WALL,
        down=PerceptionType.EMPTY,
        left=PerceptionType.WALL,
        right=PerceptionType.FOOD,
    )

    assert perception.as_tuple() == (
        PerceptionType.WALL,
        PerceptionType.EMPTY,
        PerceptionType.WALL,
        PerceptionType.FOOD,
    )


def test_equal_perceptions_are_equal():

    perception_a = Perception(
        up=PerceptionType.WALL,
        down=PerceptionType.EMPTY,
        left=PerceptionType.WALL,
        right=PerceptionType.FOOD,
    )

    perception_b = Perception(
        up=PerceptionType.WALL,
        down=PerceptionType.EMPTY,
        left=PerceptionType.WALL,
        right=PerceptionType.FOOD,
    )

    assert perception_a == perception_b


def test_perception_is_immutable():

    perception = Perception(
        up=PerceptionType.WALL,
        down=PerceptionType.EMPTY,
        left=PerceptionType.WALL,
        right=PerceptionType.FOOD,
    )

    with pytest.raises(FrozenInstanceError):
        perception.up = PerceptionType.FOOD


def test_perception_is_hashable():

    perception = Perception(
        up=PerceptionType.WALL,
        down=PerceptionType.EMPTY,
        left=PerceptionType.WALL,
        right=PerceptionType.FOOD,
    )

    assert isinstance(hash(perception), int)


def test_perception_type_contains_valid_values():

    assert PerceptionType.WALL.value == "wall"
    assert PerceptionType.EMPTY.value == "empty"
    assert PerceptionType.FOOD.value == "food"
    assert PerceptionType.OBSTACLE.value == "obstacle"


def test_perception_uses_perception_types():

    perception = Perception(
        up=PerceptionType.WALL,
        down=PerceptionType.EMPTY,
        left=PerceptionType.OBSTACLE,
        right=PerceptionType.FOOD,
    )

    assert perception.up is PerceptionType.WALL
    assert perception.down is PerceptionType.EMPTY
    assert perception.left is PerceptionType.OBSTACLE
    assert perception.right is PerceptionType.FOOD