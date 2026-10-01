import numpy as np

from ai_cat.perception.perception import Perception, PerceptionType

class Environment:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.cat_position = np.array([0, 0])

        self.foods = {
            (7, 7),
            (2, 6),
            (8, 2),
        }

        self.obstacles = {
            (3, 1),
            (4, 1),
            (5, 1),
        }

    def step(self, action):
        x, y = self.cat_position

        if action == "up" and y > 0:
            y -= 1

        elif action == "down" and y < self.size - 1:
            y += 1

        elif action == "left" and x > 0:
            x -= 1

        elif action == "right" and x < self.size - 1:
            x += 1

        new_position = (int(x), int(y))

        if new_position not in self.obstacles:
            self.cat_position = np.array(new_position)

        reward = -1
        done = False

        if tuple(self.cat_position) in self.foods:
            reward = 10
            done = True

        return reward, done

    def get_perception(self):
        x, y = self.cat_position

        positions = {
            "up": (x, y - 1),
            "down": (x, y + 1),
            "left": (x - 1, y),
            "right": (x + 1, y),
        }

        values = {}

        for direction, (px, py) in positions.items():
            if px < 0 or px >= self.size or py < 0 or py >= self.size:
                values[direction] = PerceptionType.WALL

            elif (px, py) in self.obstacles:
                values[direction] = PerceptionType.OBSTACLE

            elif (px, py) in self.foods:
                values[direction] = PerceptionType.FOOD

            else:
                values[direction] = PerceptionType.EMPTY

        return Perception(
            up=values["up"],
            down=values["down"],
            left=values["left"],
            right=values["right"],
        )

    def get_perception_state(self):
        return self.get_perception().as_tuple()

    def display(self):
        for y in range(self.size):
            row = ""

            for x in range(self.size):
                if [x, y] == list(self.cat_position):
                    row += "🐱 "
                elif (x, y) in self.foods:
                    row += "🍎 "
                elif (x, y) in self.obstacles:
                    row += "█ "
                else:
                    row += ". "

            print(row)

        print()