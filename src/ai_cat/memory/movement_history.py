from collections import deque

class MovementHistory:
    def __init__(self, capacity=12):
        if capacity <= 0:
            raise ValueError(
                "Movement history capacity must be greater than zero"
            )

        self.capacity = capacity
        self._positions = deque(maxlen=capacity)

    def add(self, position):
        self._positions.append(tuple(position))

    def get(self):
        return tuple(self._positions)

    def reset(self):
        self._positions.clear()

    def has_cycle(self, cycle_length=2):
        if cycle_length <= 0:
            raise ValueError(
                "Cycle length must be greater than zero"
            )

        positions = self.get()

        if len(positions) < cycle_length * 2:
            return False

        return (
            positions[-cycle_length:]
            == positions[-cycle_length * 2:-cycle_length]
        )
