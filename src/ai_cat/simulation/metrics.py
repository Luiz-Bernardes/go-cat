class SimulationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.foods_found = 0
        self.cycles_detected = 0
        self.visited_positions = set()
        self.cycle_events = 0

    def record_cycle_event(self):
        self.cycle_events += 1

    def record_position(self, position):
        self.visited_positions.add(tuple(position))

    def record_step(self):
        self.steps += 1

    def record_food(self):
        self.foods_found += 1

    def record_cycle(self):
        self.cycles_detected += 1

    @property
    def unique_positions(self):
        return len(self.visited_positions)

    def summary(self):
        return {
            "steps": self.steps,
            "unique_positions": self.unique_positions,
            "foods_found": self.foods_found,
            "cycles_detected": self.cycles_detected,
            "cycle_events": self.cycle_events,
        }
