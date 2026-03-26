# Furuta
import torch
import numpy as np
import copy
import sys
import os
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


repo_path = os.path.join(script_dir, "vision-based-furuta-pendulum")
print("Repo exists:", os.path.exists(repo_path))

sys.path.insert(0, repo_path)
# from vision_based_furuta_pendulum_master.gym_brt.envs import QubeBalanceEnv, QubeSwingupEnv
# from vision_based_furuta_pendulum_master.gym_brt.control.control import QubeHoldControl, QubeFlipUpControl

from gym_brt.envs import QubeBalanceEnv, QubeSwingupEnv
from gym_brt.control.control import QubeHoldControl, QubeFlipUpControl
from IPython import embed as IPS



class ground_truth_Furuta():
    def __init__(self, safety_threshold, use_simulator):
        self.use_simulator = use_simulator
        self.safety_threshold = safety_threshold
        self.frequency = 200
        self.freq_div = 4
        self.k_scale = np.diag([-10, 100, 1, 1])  # scale it a priori.
        self.last_two_entries = np.array([-1.5040040945983464, 3.0344775662414483])  # these are kept constant
        with QubeBalanceEnv(use_simulator=use_simulator, frequency=self.frequency) as env:
            self.state_init = env.reset()

    def conduct_experiment(self, x, noise_std=None):
        param = np.asarray(x, dtype=np.float64).flatten()
        param = np.concatenate((param, self.last_two_entries))
        self.state = copy.deepcopy(self.state_init)
        reward = 0
        if not self.use_simulator:
            IPS()
        with QubeSwingupEnv(use_simulator=self.use_simulator, frequency=self.frequency) as env:
            env.reset()
            swing_up_ctrl = QubeFlipUpControl(sample_freq=self.frequency, env=env)
            upright = False
            i = 0
            theta_max = -np.inf
            alpha_max = -np.inf
            theta_list = []
            alpha_list = []
            while i < 1000:
                if upright:
                    if np.abs(self.state[1]) < math.pi/2 and i % self.freq_div == 0:
                        action = np.dot(np.dot(self.k_scale, param.flatten()), self.state)
                    elif np.abs(self.state[1]) >= math.pi/2:
                        action = np.array([0.0])
                        # print("failed during regular SafeOpt")  # why though?
                    self.state, rew, _, _ = env.step(action.flatten())
                    reward += rew
                    i += 1
                    # print(self.state)
                    theta, alpha, theta_dot, alpha_dot = self.state
                    theta_list.append(theta)
                    alpha_list.append(alpha)
                    if np.abs(theta) > theta_max:
                        theta_max = np.abs(theta)
                    if np.abs(alpha) > alpha_max:
                        alpha_max = np.abs(alpha)
                else:
                    action = swing_up_ctrl.action(self.state)*1.4
                    self.state, _, _, _ = env.step(action)
                    if not self.use_simulator:
                        pass
                        print(np.linalg.norm(self.state))
                    if np.linalg.norm(self.state) < 8e-2:
                        print("swingup completed")
                        upright = True
            constraint_1 = math.pi/2 - theta_max    # - math.pi/2, theta_max - math.pi/2)
            constraint_2 = math.pi/4 - alpha_max  #  also works. But this is safer if optimization is possible.
            # Box angle up to 80 degrees
            # Other angle up to 10 degrees
            print(f'Experiment done. For the parameter {x}, we received the reward {(reward/1000)} and the min constraint value {min(constraint_1,constraint_2)}')
        return torch.tensor([reward/1000], dtype=torch.float32), torch.tensor([constraint_1],dtype=torch.float32), torch.tensor([constraint_2],dtype=torch.float32), alpha_list, theta_list

    def try_furuta_real(self, param):
        param = np.asarray(param, dtype=np.float64).flatten()
        last_two_entries = self.last_two_entries
        param = np.concatenate((param, last_two_entries))
        use_simulator = self.use_simulator
        frequency = 200
        divider = 4
        constr = np.inf
        with QubeSwingupEnv(use_simulator=use_simulator, frequency=frequency) as env:
            state = env.reset()
            swing_up_ctrl = QubeFlipUpControl(sample_freq=frequency, env=env)
            upright = False
            i = 0
            reward = 0
            reward_LQR = 0
            R = 1
            Q = np.diag([5, 1, 1, 1])
            while i < 1000:
                if upright:
                    if np.abs(state[1]) < math.pi/2 and i % divider == 0:
                        action = np.dot(np.dot(self.k_scale, param), state)  # directly scale
                    elif np.abs(state[1]) >= math.pi/2:
                        action = np.array([0.0])
                        raise Exception("failed")
                    state, rew, _, _ = env.step(action.flatten())
                    reward += rew
                    reward_LQR += -(state.T@Q@state + (action**2*R))
                    dist_constr = [np.pi/2 - np.abs(state[idx]) for idx in range(2)]
                    if np.min(dist_constr) < constr:
                        constr = np.min(dist_constr)
                    i += 1
                else:
                    action = swing_up_ctrl.action(state)*1.4
                    state, _, _, _ = env.step(action)
                    print(np.linalg.norm(state))
                    if np.linalg.norm(state) < 8e-2:  # 5e-2:
                        upright = True
        return torch.tensor([reward/1000], dtype=torch.float32)