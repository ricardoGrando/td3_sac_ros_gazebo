#!/usr/bin/env python

import os
import numpy as np
from td3_agent import TD3Agent
# from utils_new import plot_learning_curve, make_env
import rospy
from environment import *
import time

def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])

    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    rospy.init_node('deep_rl_gazebo')

    n_games = rospy.get_param('~n_games')
    ep_start = rospy.get_param('~ep_start')
    lr = rospy.get_param('~lr')
    max_mem = rospy.get_param('~max_mem')
    bs = rospy.get_param('~bs')
    gpu = rospy.get_param('~gpu')
    load_checkpoints = rospy.get_param('~load_checkpoints')
    path = rospy.get_param('~path')
    state_dim = int(rospy.get_param('~state_dim'))
    action_dim = int(rospy.get_param('~action_dim'))
    norm_value = float(rospy.get_param('~norm_value'))
    laser_samples = int(rospy.get_param('~laser_samples'))
    env_name = rospy.get_param('~env_name')
    max_action = float(rospy.get_param('~max_action'))
    min_action = float(rospy.get_param('~min_action'))
    max_steps = 500
    alpha = 0.001
    beta = 0.001
    tau = 0.005

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    env = Env(env_name=env_name, state_dim=state_dim, action_dim=action_dim, norm_value=norm_value, \
                    max_steps=max_steps)

    best_score = -np.inf

    agent = TD3Agent(alpha=alpha, beta=beta, input_dims=env.observation_space, tau=tau, max_action=max_action, \
                        min_action=min_action, batch_size=bs, layer1_size=512, layer2_size=512, n_actions=env.action.shape[0], \
                        path_dir=path)

    ep = ep_start
    if load_checkpoints:
        agent.load_models(ep)

    score_history = []

    #agent.load_models()

    for i in range(ep_start, n_games):
        observation = env.reset()
        done = False
        score = 0

        while True:
             action = agent.choose_action(observation)
             # past_action = observation[laser_samples:laser_samples+action_dim]
             observation_, reward, done = env.step(action)
             # print(observation_)
             agent.remember(observation, action, reward, observation_, done)
             agent.learn()
             score += reward
             observation = observation_
             if done: break
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if ep % 20 == 0:
            # best_score = avg_score
            agent.save_models(ep)

        print('Episode', i, 'score %.1f' % score, 'average score %.1f' % avg_score)
        print('Target goal: %.2f' % env.goal_x, ' %.2f' % env.goal_y)

        ep += 1

    # x = [i+1 for i in range(n_games)]
    # plot_learning_curve(x, score_history, figure_file)
