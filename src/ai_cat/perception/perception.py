class Perception:
    def __init__(self, up, down, left, right):
        self.up = up
        self.down = down
        self.left = left
        self.right = right

    def as_tuple(self):
        return (
            self.up,
            self.down,
            self.left,
            self.right,
        )

    def __eq__(self, other):
        if not isinstance(other, Perception):
            return NotImplemented

        return self.as_tuple() == other.as_tuple()

    def __repr__(self):
        return (
            f"Perception("
            f"up={self.up!r}, "
            f"down={self.down!r}, "
            f"left={self.left!r}, "
            f"right={self.right!r}"
            f")"
        )