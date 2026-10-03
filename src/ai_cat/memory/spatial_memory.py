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