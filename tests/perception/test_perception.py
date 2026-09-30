from ai_cat.perception.perception import Perception


def test_perception_stores_directions():

    perception = Perception(
        up="wall",
        down="empty",
        left="wall",
        right="food",
    )

    assert perception.up == "wall"
    assert perception.down == "empty"
    assert perception.left == "wall"
    assert perception.right == "food"


def test_perception_can_be_converted_to_tuple():

    perception = Perception(
        up="wall",
        down="empty",
        left="wall",
        right="food",
    )

    assert perception.as_tuple() == (
        "wall",
        "empty",
        "wall",
        "food",
    )


def test_equal_perceptions_are_equal():

    perception_a = Perception(
        up="wall",
        down="empty",
        left="wall",
        right="food",
    )

    perception_b = Perception(
        up="wall",
        down="empty",
        left="wall",
        right="food",
    )

    assert perception_a == perception_b