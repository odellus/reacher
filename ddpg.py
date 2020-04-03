# -*- coding: utf-8
import sys
import os
import gym
import random
import torch
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pickle as pkl

from collections import deque

from ddpg_agent import Agent
from utils import load_cfg, get_state_action_sizes, pkl_dump, yaml_dump



cfg = load_cfg()

N_EPISODES = cfg["Training"]["Number_episodes"]
MAX_TIMESTEPS = cfg["Training"]["Max_timesteps"]
PRINT_EVERY = cfg["Training"]["Score_window"]

FPATH = cfg["Environment"]["Filepath"]
UNITY_PYTHONPATH = cfg["Environment"]["Unity_pythonpath"]

BRAIN_INDEX = cfg["Agent"]["Brain_index"]
RANDOM_SEED = cfg["Environment"]["Random_seed"]

def get_agent_unity():
    sys.path.append(UNITY_PYTHONPATH)
    from unityagents import UnityEnvironment
    env = UnityEnvironment(file_name=FPATH, seed=RANDOM_SEED)
    brain_name = env.brain_names[BRAIN_INDEX]
    brain = env.brains[brain_name]
    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=RANDOM_SEED)
    return env, agent

env, agent = get_agent_unity()

BRAIN_NAME = env.brain_names[BRAIN_INDEX]

def step_unity(
    env,
    action,
    brain_index=BRAIN_INDEX,
    brain_name=BRAIN_NAME
    ):
    """Step Unity environment forward one timestep

    Params
    ======
        env (UnityEnvironment): The Unity environment to step forwards
        action (int): The action index to take during this timestep
        brain_index (int): The brain index of the agent we wish to act
        brain_name (str): The name of the brain we wish to act
    """
    env_info = env.step(action)[brain_name]
    state = env_info.vector_observations[brain_index]
    reward = env_info.rewards[brain_index]
    done = env_info.local_done[brain_index]
    return state, reward, done, env_info



def get_agent_gym():
    env = gym.make("Pendulum-v0")
    env.seed(2)

    state_size, action_size = get_state_action_sizes(env)
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)
    return env, agent

def setup_experiment():
    t_experiment = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment_dir = "experiment_{}".format(t_experiment)
    os.mkdir(experiment_dir)
    yaml_dump(cfg, "./{}/config.yaml".format(experiment_dir))
    return experiment_dir

def persist_experiment(t_experiment, i_episode, agent, scores):
    os.chdir(t_experiment)
    torch.save(agent.actor_local.state_dict(), "checkpoint_actor_{}.pth".format(i_episode))
    torch.save(agent.critic_local.state_dict(), "checkpoint_critic_{}.pth".format(i_episode))
    pkl_dump(scores, "scores_{}.pkl".format(i_episode))
    os.chdir("..")

def load_pretrained(agent):
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor_2100.pth'))
    agent.actor_target.load_state_dict(torch.load('checkpoint_actor_2100.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic_2100.pth'))
    agent.critic_target.load_state_dict(torch.load('checkpoint_critic_2100.pth'))
    return agent

def ddpg(
    n_episodes=N_EPISODES,
    max_t=MAX_TIMESTEPS,
    print_every=PRINT_EVERY
    ):
    scores_deque = deque(maxlen=print_every)
    scores = []
    # Create a directory to save the findings.
    experiment_dir = setup_experiment()
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[BRAIN_NAME]
        state = env_info.vector_observations[BRAIN_INDEX]
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = step_unity(env, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)), end="")
        visualize = False
        if i_episode % print_every == 0:
            persist_experiment(experiment_dir, i_episode, agent, scores)
            print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))
            print("\rEpisode {}\tStandard Deviation of Last {} Scores: {:.2f}".format(i_episode, print_every, np.std(scores_deque)))


    return scores

if __name__ == "__main__":
    agent = load_pretrained(agent)
    scores = ddpg()
