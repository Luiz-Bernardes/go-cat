import numpy as np

class Environment:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.cat_position = np.array([0, 0])
        self.food_position = np.array([7, 7])

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

        self.cat_position = np.array([int(x), int(y)])

        reward = -1
        done = False

        if np.array_equal(self.cat_position, self.food_position):
            reward = 10
            done = True

        return reward, done

    def display(self):
        for y in range(self.size):
            row = ""

            for x in range(self.size):
                if [x, y] == list(self.cat_position):
                    row += "🐱 "
                elif [x, y] == list(self.food_position):
                    row += "🍎 "
                else:
                    row += ". "

            print(row)

        print()