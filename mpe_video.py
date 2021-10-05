import cv2  # pip3 install opencv-python
# import gym  # pip3 install gym==0.17 pyglet==1.5.0  # env.render() bug in gym==0.18, pyglet==1.6
import torch

import os
import gym  # not necessary
import numpy as np

from elegantrl.run import *
from elegantrl.agent import AgentDQN
from elegantrl.env import PreprocessEnv

"""init env"""
env = mpe_make_env('simple_spread')

'''init agent'''
# agent = None   # means use random action
agent = AgentDQN()  # means use the policy network which saved in cwd
agent_cwd = 'elegantRL-marl\AgentDQN_simple_spread_0'
net_dim = 2 ** 8
state_dim = env.state_dim
action_dim = env.action_dim
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

agent.init(net_dim, state_dim, action_dim)
agent.save_or_load_agent(cwd=agent_cwd, if_save=False)
device = agent.device
assert 0
'''initialize evaluete and env.render()'''
save_frame_dir = ''  # means don't save video, just open the env.render()
# save_frame_dir = 'frames'  # means save video in this directory
if save_frame_dir:
    os.makedirs(save_frame_dir, exist_ok=True)

state = env.reset()
episode_return = 0
step = 0
for i in range(2 ** 10):
    print(i) if i % 128 == 0 else None
    for j in range(1):
        if agent is None:
            action = env.action_space.sample()
        else:
            s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=device)
            a_tensor = agent.act(s_tensor)
            action = a_tensor.detach().cpu().numpy()[0]  # if use 'with torch.no_grad()', then '.detach()' not need.
        next_state, reward, done, _ = env.step(action)

        episode_return += reward
        step += 1

        if done:
            print(f'\t'
                    f'TotalStep {i:>6}, epiStep {step:6.0f}, '
                    f'Reward_T {reward:8.3f}, epiReward {episode_return:8.3f}')
            state = env.reset()
            episode_return = 0
            step = 0
        else:
            state = next_state

    if save_frame_dir:
        frame = env.render('rgb_array')
        cv2.imwrite(f'{save_frame_dir}/{i:06}.png', frame)
        cv2.imshow('OpenCV Window', frame)
        cv2.waitKey(1)
    else:
        env.render()
env.close()

'''convert frames png/jpg to video mp4/avi using ffmpeg'''
if save_frame_dir:
    frame_shape = cv2.imread(f'{save_frame_dir}/{3:06}.png').shape
    print(f"frame_shape: {frame_shape}")

    save_video = 'gym_render.mp4'
    os.system(f"| Convert frames to video using ffmpeg. Save in {save_video}")
    os.system(f'ffmpeg -r 60 -f image2 -s {frame_shape[0]}x{frame_shape[1]} '
                f'-i ./{save_frame_dir}/%06d.png '
                f'-crf 25 -vb 20M -pix_fmt yuv420p {save_video}')
