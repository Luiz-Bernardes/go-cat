class State:
    UNKNOWN = (
        "unknown",
        "unknown",
        "unknown",
        "unknown",
    )

    def __init__(self, memory, capacity):
        if capacity <= 0:
            raise ValueError("State capacity must be greater than zero")

        self.memory = memory
        self.capacity = capacity

    def build(self):
        perceptions = self.memory.get()

        states = [
            self.perception_to_state(perception)
            for perception in perceptions[-self.capacity:]
        ]

        padding = [
            self.UNKNOWN
        ] * (self.capacity - len(states))

        return tuple(padding + states)

    @staticmethod
    def perception_to_state(perception):
        return perception.as_tuple()