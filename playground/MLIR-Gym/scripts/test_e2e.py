import random
import numpy as np

def sanity_run(env, seed, rounds):
    random.seed(seed)
    np.random.seed(seed)

    t_obs, t_act, t_reward = [], [], []

    # run episode for this mlir file
    for episode_num in range(0, rounds):
        obs, info = env.reset()
        over = False
        while not over:
            act = env.action_space.sample()
            print(env.action_layer.idx_to_action(act))
            obs, reward, term, trun, info = env.step(act)
            print(reward)

            t_obs.append(obs)
            t_act.append(act)
            t_reward.append(reward)

            over = term or trun

            # end of episode
            print(f"Episode: {episode_num}")
            print(f"total rewards: {np.sum(t_reward)}")

            trajectories = []
            trajectories.append(t_obs)
            trajectories.append(t_act)
            trajectories.append(t_reward)
            t_obs, t_act, t_reward = [], [], []

    env.close()
    print("env close")

