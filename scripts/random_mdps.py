import numpy as np
import gym
from finite_mdp.envs.finite_mdp import DeterministicMDP, FiniteMDPEnv
from rl_agents.agents.dynamic_programming.robust_value_iteration import RobustValueIterationAgent
from rl_agents.agents.simple.random import RandomUniformAgent
from rl_agents.agents.tree_search.robust_mcts import DiscreteRobustMCTSAgent
from rl_agents.trainer.evaluation import Evaluation


def evaluate(e, a):
    state = e.reset()
    done = False
    total_reward = 0
    while not done:
        state, reward, done, info = e.step(a.act(state))
        total_reward += reward
    return total_reward


def main(episodes=3):
    gym.logger.set_level(gym.logger.INFO)
    FiniteMDPEnv.MAX_STEPS = 5
    size = (4, 2)
    actions = []
    for _ in range(episodes):
        # MDPs
        m1 = DeterministicMDP(transition=np.zeros(size), reward=np.zeros(size))
        m1.randomize()
        m2 = DeterministicMDP(transition=np.zeros(size), reward=np.zeros(size))
        m2.randomize()

        # Environment
        for m in [m1, m2]:
            env = gym.make("finite-mdp-v0")
            env.configure(m.to_config())
            state = env.reset()

            # Agents
            models = [m1.to_config(), m2.to_config()]
            preprocessors = [[{"method": "copy_with_config", "args": m}] for m in models]
            rvi = RobustValueIterationAgent(env, config={"models": models,
                                                         "iterations": 5})
            rmcts = DiscreteRobustMCTSAgent(env, config={"envs_preprocessors": preprocessors,
                                                         "iterations": 5000,
                                                         "temperature:": 500,
                                                         "step_strategy": "subtree",
                                                         "max_depth": 5})
            rand = RandomUniformAgent(env)

            print("RVI value", rvi.state_value()[state])
            # done = False
            # local_actions = []
            # total_reward = 0
            # while not done:
            #     a_rvi = rvi.act(state)
            #     a_rmcts = rmcts.act(state)
            #     state, reward, done, info = env.step(a_rvi)
            #     local_actions.append([a_rvi, a_rmcts])
            #     total_reward += reward
            # actions += local_actions
            # local_actions = np.array(local_actions)
            # print(np.count_nonzero(local_actions[:, 0] - local_actions[:, 1]), "out of", np.shape(local_actions)[0])

            print("total reward RVI", evaluate(env, rvi))
            print("total reward RMCTS", evaluate(env, rmcts))
            print("total reward Random", evaluate(env, rand))

    # actions = np.array(actions)
    # print(np.count_nonzero(actions[:, 0] - actions[:, 1]), "out of", np.shape(actions)[0])


if __name__ == "__main__":
    main()
