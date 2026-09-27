from ai_cat.environments.grid_world import Environment
from ai_cat.agents.agent import Agent

def create_ambiguous_environment():
    environment = Environment(size=7)

    environment.foods = {
        (6, 1),
        (0, 5),
    }

    environment.obstacles = {
        # Linha superior
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),

        # Linha inferior
        (1, 6),
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
    }

    return environment


def train_agent(agent, environment):
    episodes = 2000
    max_steps = 100

    for _ in range(episodes):
        environment.reset()

        for _ in range(max_steps):
            state = environment.get_perception_state()

            action = agent.choose_action(state)

            reward, done = environment.step(action)

            next_state = environment.get_perception_state()

            agent.learn(
                state,
                action,
                reward,
                next_state,
            )

            if done:
                break

        agent.decay_exploration()


def run_agent_from_position(agent, environment, position):
    environment.reset()

    environment.cat_position = list(position)

    agent.learning.exploration_rate = 0

    path = []

    for _ in range(30):
        state = environment.get_perception_state()

        action = agent.choose_action(state)

        reward, done = environment.step(action)

        path.append(
            (
                action,
                tuple(environment.cat_position),
                reward,
            )
        )

        if done:
            return True, path

    return False, path


def test_same_perception_can_require_different_actions():
    environment = create_ambiguous_environment()
    agent = Agent()

    train_agent(
        agent,
        environment,
    )

    position_a = (2, 2)
    position_b = (4, 4)

    environment.cat_position = list(position_a)
    perception_a = environment.get_perception_state()

    environment.cat_position = list(position_b)
    perception_b = environment.get_perception_state()

    print()
    print("Percepção A:", perception_a)
    print("Percepção B:", perception_b)

    assert perception_a == perception_b

    success_a, path_a = run_agent_from_position(
        agent,
        environment,
        position_a,
    )

    success_b, path_b = run_agent_from_position(
        agent,
        environment,
        position_b,
    )

    print()
    print("Posição A:", position_a)
    print("Caminho A:")

    for step, (action, position, reward) in enumerate(path_a, 1):
        print(
            f"{step:02d} | "
            f"{action:5s} | "
            f"{position} | "
            f"reward={reward}"
        )

    print()
    print("Posição B:", position_b)
    print("Caminho B:")

    for step, (action, position, reward) in enumerate(path_b, 1):
        print(
            f"{step:02d} | "
            f"{action:5s} | "
            f"{position} | "
            f"reward={reward}"
        )

    print()
    print("Sucesso A:", success_a)
    print("Sucesso B:", success_b)

    assert success_a
    assert success_b