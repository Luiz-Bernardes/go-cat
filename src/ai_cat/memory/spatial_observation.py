from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialObservation:
    up_visited: bool
    down_visited: bool
    left_visited: bool
    right_visited: bool

    def as_tuple(self):
        return (
            self.up_visited,
            self.down_visited,
            self.left_visited,
            self.right_visited,
        )