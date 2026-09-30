from dataclasses import dataclass


@dataclass(frozen=True)
class Perception:
    up: str
    down: str
    left: str
    right: str

    def as_tuple(self):
        return (
            self.up,
            self.down,
            self.left,
            self.right,
        )
