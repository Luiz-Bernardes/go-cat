class SpatialMemory:
    def __init__(self):
        self._positions = set()

    def add(self, position):
        self._positions.add(tuple(position))

    def contains(self, position):
        return tuple(position) in self._positions

    def get(self):
        return tuple(self._positions)

    def reset(self):
        self._positions.clear()

    def __len__(self):
        return len(self._positions)

    def observe_position(self, position):
        x, y = position

        positions = {
            "up": (x, y - 1),
            "down": (x, y + 1),
            "left": (x - 1, y),
            "right": (x + 1, y),
        }

        return {
            direction: self.contains(neighbor)
            for direction, neighbor in positions.items()
        }