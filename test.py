from elegantrl.run import *
from elegantrl.agent import AgentDQN
from elegantrl.env import PreprocessEnv
import gym
gym.logger.set_level(40) # Block warning    
env = mpe_make_env('simple_spread')
args = Arguments(if_on_policy=False)
args.agent = AgentDQN()  # AgentSAC(), AgentTD3(), AgentDDPG()
args.env = PreprocessEnv(env)
args.reward_scale = 2 ** -1  # RewardRange: -200 < -150 < 300 < 334
args.gamma = 0.95
args.rollout_num = 2# the number of rollout workers (larger is not always faster)
train_and_evaluate(args) # the training process will terminate once it reaches the target reward.