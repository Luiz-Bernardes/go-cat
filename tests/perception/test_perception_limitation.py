from ai_cat.environments.grid_world import Environment
from ai_cat.agents.agent import Agent

def create_environment(start_position, food_position):
    environment = Environment(size=7)

    environment.cat_position = list(start_position)

    environment.foods = {
        food_position,
    }

    environment.obstacles = set()

    return environment


def train_agent(agent, environment):
    episodes = 500
    max_steps = 30

    for _ in range(episodes):
        environment.cat_position = [2, 3]

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


def test_same_perception_requires_opposite_actions():
    # Ambiente A:
    # gato precisa ir para a direita.
    environment_a = create_environment(
        start_position=(2, 3),
        food_position=(6, 3),
    )

    # Ambiente B:
    # gato precisa ir para a esquerda.
    environment_b = create_environment(
        start_position=(4, 3),
        food_position=(0, 3),
    )

    agent = Agent()

    train_agent(
        agent,
        environment_a,
    )

    agent.learning.exploration_rate = 0

    environment_a.cat_position = [2, 3]
    environment_b.cat_position = [4, 3]

    perception_a = environment_a.get_perception_state()
    perception_b = environment_b.get_perception_state()

    print()
    print("Percepção A:", perception_a)
    print("Percepção B:", perception_b)

    assert perception_a == perception_b

    action_a = agent.choose_action(perception_a)
    action_b = agent.choose_action(perception_b)

    print()
    print("Ação no ambiente A:", action_a)
    print("Ação no ambiente B:", action_b)

    print()
    print("Ação correta em A: right")
    print("Ação correta em B: left")

    # Como as duas situações possuem exatamente a mesma percepção,
    # o agente recebe o mesmo estado e não consegue diferenciá-las.
    assert action_a == action_b