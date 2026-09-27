from ai_cat.environments.grid_world import Environment
from ai_cat.agents.agent import Agent

def create_training_environment():
    environment = Environment()

    environment.foods = {
        (8, 2),
        (2, 6),
        (7, 7),
    }

    environment.obstacles = {
        (3, 1),
        (4, 1),
        (5, 1),
    }

    return environment


def create_test_environment():
    environment = Environment()

    environment.foods = {
        (8, 3),
        (4, 5),
    }

    environment.obstacles = {
        (5, 1),
        (6, 1),
        (2, 3),
        (3, 3),
    }

    return environment


def train_agent(agent, environment):
    episodes = 1000
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


def run_test(agent, environment):
    agent.learning.exploration_rate = 0

    max_steps = 100
    steps = 0

    environment.reset()

    for step in range(max_steps):
        state = environment.get_perception_state()

        action = agent.choose_action(state)

        reward, done = environment.step(action)

        steps = step + 1

        if done:
            return True, steps

    return False, steps


def test_agent_generalizes_from_one_environment_to_another():
    training_environment = create_training_environment()
    test_environment = create_test_environment()

    agent = Agent()

    train_agent(
        agent,
        training_environment,
    )

    successes = 0
    total_steps = 0
    test_runs = 100

    for _ in range(test_runs):
        success, steps = run_test(
            agent,
            test_environment,
        )

        if success:
            successes += 1
            total_steps += steps

    print()
    print("Generalização entre ambientes")
    print(f"Testes: {test_runs}")
    print(f"Sucessos: {successes}")
    print(f"Falhas: {test_runs - successes}")

    if successes:
        print(
            f"Média de passos: "
            f"{total_steps / successes:.2f}"
        )

    assert successes > 0