from collections import deque

class Memory:
    def __init__(self, capacity=3):
        if capacity <= 0:
            raise ValueError("Memory capacity must be greater than zero")

        self.capacity = capacity
        self._items = deque(maxlen=capacity)

    def add(self, item):
        self._items.append(item)

    def get(self):
        return tuple(self._items)

    def reset(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)