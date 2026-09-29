from ai_cat.environments.grid_world import Environment
from ai_cat.agents.agent import Agent

def configure_experiment_environment(environment):
    environment.foods = {
        (4, 0),
        (0, 5),
    }

    environment.obstacles = {
        (1, 0),
        (2, 0),
        (3, 0),
    }

def test_trained_agent_finds_food_without_crossing_obstacles():
    environment = Environment()
    agent = Agent()

    environment.foods = {
        (4, 0),
        (0, 8),
    }

    environment.obstacles = {
        (1, 0),
        (2, 0),
        (3, 0),
    }

    episodes = 1000
    max_steps = 100

    # Treinamento
    for _ in range(episodes):
        environment.reset()
        configure_experiment_environment(environment)

        agent.position = [0, 0]
        agent.memory.reset()

        # Primeira percepção do episódio.
        agent.observe(environment)

        for _ in range(max_steps):
            state = agent.get_state()

            action = agent.choose_action(state)

            reward, done = environment.step(action)

            agent.position = list(environment.cat_position)

            # Nova percepção após a ação.
            agent.observe(environment)

            next_state = agent.get_state()

            agent.learn(
                state,
                action,
                reward,
                next_state,
            )

            if done:
                break

        agent.decay_exploration()

    # Desliga exploração
    agent.learning.exploration_rate = 0

    environment.reset()
    configure_experiment_environment(environment)

    agent.position = [0, 0]
    agent.memory.reset()

    # Primeira percepção da avaliação.
    agent.observe(environment)

    print("\nCaminho aprendido:\n")

    for step in range(max_steps):
        state = agent.get_state()

        action = agent.choose_action(state)

        reward, done = environment.step(action)

        agent.position = list(environment.cat_position)

        print(
            f"{step + 1:02d} | "
            f"{action:5} | "
            f"{agent.position}"
        )

        if done:
            break

        # Observa o ambiente após a ação.
        agent.observe(environment)

    print("\nResultado:")
    print(f"Comida encontrada: {tuple(agent.position)}")
    print(f"Passos: {step + 1}")
    print(f"Recompensa: {reward}")
    print("\nMapa final:\n")
    environment.display()

if __name__ == "__main__":
    test_trained_agent_finds_food_without_crossing_obstacles()