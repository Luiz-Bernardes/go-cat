import numpy as np


class Environment:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.cat_position = np.array([0, 0])

        self.foods = {
            (7, 7),
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