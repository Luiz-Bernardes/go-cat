class State:
    UNKNOWN = (
        "unknown",
        "unknown",
        "unknown",
        "unknown",
    )

    def __init__(self, memory, capacity):
        self.memory = memory
        self.capacity = capacity

    def build(self):
        states = [
            self.perception_to_state(perception)
            for perception in self.memory.get()
        ]

        padding = [
            self.UNKNOWN
        ] * (self.capacity - len(states))

        return tuple(padding + states)

    @staticmethod
    def perception_to_state(perception):
        return (
            perception["up"],
            perception["down"],
            perception["left"],
            perception["right"],
        )