from dataclasses import dataclass
from enum import Enum


class PerceptionType(Enum):
    WALL = "wall"
    EMPTY = "empty"
    FOOD = "food"
    OBSTACLE = "obstacle"


@dataclass(frozen=True)
class Perception:
    up: PerceptionType
    down: PerceptionType
    left: PerceptionType
    right: PerceptionType

    def as_tuple(self):
        return (
            self.up,
            self.down,
            self.left,
            self.right,
        )