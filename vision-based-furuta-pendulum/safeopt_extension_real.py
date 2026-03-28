from __future__ import print_function, division, absolute_import
import GPy
import numpy as np 
import math
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import copy 
import pickle
import sys
sys.path.insert(1,  './..')
import my_safeopt
import safeopt
import gym
from gym_brt.envs import QubeBalanceEnv, QubeSwingupEnv
from gym_brt.control.control import QubeHoldControl, QubeFlipUpControl
from IPython import embed as IPS
import warnings


def save_gps(filename1, filename2, gp, gp_constr):
	with open(filename1, 'wb') as output:
		pickle.dump(gp, output, pickle.HIGHEST_PROTOCOL)
	with open(filename2, 'wb') as output:
		pickle.dump(gp_constr, output, pickle.HIGHEST_PROTOCOL)

class safe_opt:

	def __init__(self,initial_param=np.array([[0.9, 0.3, -1.5040040945983464, 3.0344775662414483]]),noise_var=0.01**2,bounds_param=[(0,1),(0,1)],threshold=1e-4,sim=True):
		self.param, self.noise_var, self.bounds_param, self.threshold, self.sim = initial_param, noise_var, bounds_param, threshold, sim
		high_obs = np.array([math.pi/2,math.pi/2,np.inf,np.inf])
		self.state_bounds = list(zip(-high_obs,high_obs))
		self.kernel = GPy.kern.sde_Matern32(input_dim=len(self.bounds_param), variance=1, lengthscale=.1, ARD=True)
		self.kernel_constr = self.kernel.copy()
		self.frequency = 200  
		self.freq_div = 4
		with QubeBalanceEnv(use_simulator=True, frequency=self.frequency) as env:
			self.state_init = env.reset()
		self.state = copy.deepcopy(self.state_init)
		self.k_scale = np.diag([-10, 100, 1, 1])
		self.k_opt = copy.deepcopy(initial_param)
		# set of parameters
		# self.parameter_set = my_safeopt.linearly_spaced_combinations(self.bounds_param, 2000)
		self.parameter_set = safeopt.linearly_spaced_combinations(self.bounds_param, 2000)
		# self.parameter_roa_set = safeopt.linearly_spaced_combinations(self.state_bounds, 100)

	def setup_optimization(self):
		self.gp = GPy.models.GPRegression(self.param[0,0:2].reshape(1,-1),self.obj_fun(self.param[0,0:2].reshape(1,-1))[:,0,None], self.kernel, noise_var=self.noise_var)
		self.gp_constr = GPy.models.GPRegression(self.param[0,0:2].reshape(1,-1),self.obj_fun(self.param[0,0:2].reshape(1,-1))[:,1,None], self.kernel_constr, noise_var=self.noise_var)
		# self.opt = safeopt.SafeOptSwarm([self.gp,self.gp_constr], [-np.inf,0.1],self.bounds_param, swarm_size=10000)
		self.opt = safeopt.SafeOpt([self.gp,self.gp_constr],self.parameter_set,[-np.inf,0],beta=3)		# why do we have 2 GPs?
		# self.opt = my_safeopt.SafeOpt(gp=[self.gp,self.gp_constr], parameter_set=self.parameter_set, fmin=[-np.inf,0], B=5, sigma_noise=np.sqrt(self.noise_var), delta=0.9, threshold=self.threshold)  
		# fmin is 2-dimensional? how... I thought just 1 reward 
		# I think that's why there are 2 GPs-> for 2 outputs; we just need one!


	def optimization(self, num_it=10, second_round=False):

		for _ in range(num_it):
			# Obtain next query point
			x_next = self.opt.optimize()
			# Get a measurement from the real system
			y_meas = self.obj_fun(x_next)
			# Add this to the GP model
			self.opt.add_new_data_point(x_next, y_meas)
		k_new, opt_new = self.opt.get_maximum()
		print('current optimum:', k_new)

	def obj_fun(self,param):
		param = np.append(param, self.k_opt[0,2::])
		self.state = copy.deepcopy(self.state_init)
		reward = 0
		constr = np.inf
		IPS()
		with QubeSwingupEnv(use_simulator=self.sim, frequency=self.frequency) as env:
			env.reset()
			swing_up_ctrl = QubeFlipUpControl(sample_freq=self.frequency, env=env)
			upright = False
			i = 0
			while i < 1000:
				if upright:
					if np.abs(self.state[1]) < math.pi/2 and i % self.freq_div == 0:
						action = np.dot(np.dot(self.k_scale, param.flatten()),self.state)
					elif np.abs(self.state[1]) >= math.pi/2:
						action = np.array([0.0])
						print("failed during regular SafeOpt")
					self.state, rew, _, _ = env.step(action.flatten())
					reward += rew
					i += 1
					dist_constr = [bound[1] - np.abs(self.state[idx]) for idx,bound in enumerate(self.state_bounds)]
					if np.min(dist_constr)<constr:
						constr = np.min(dist_constr)
				else:
					action = swing_up_ctrl.action(self.state)
					self.state, _, _, _ = env.step(action)
					if np.linalg.norm(self.state) < 1e-2:
						print("swingup completed")
						upright = True 
		if constr < 0:
			print("violated during regular SafeOpt")
		return np.array([[reward/1000,constr]])

def try_qube_env():
	num_episodes = 10
	num_steps = 250
	with qube(use_simulator = True, frequency = 500) as env:
		for episode in range(num_episodes):
			state = env.reset()
			for step in range(num_steps):
				action = env.action_space.sample()
				state, reward, done, _ = env.step(action.flatten())

def plot_save(gp, gp_constr, bounds, num_samples=1000):
	# cont_rew, colorbar_rew, data_rew = my_safeopt.utilities.plot_contour_gp(gp, [np.linspace(bounds[0][0], bounds[0][1],num_samples),np.linspace(bounds[1][0],bounds[1][1],num_samples)])
	cont_rew, colorbar_rew, data_rew = safeopt.utilities.plot_contour_gp(gp, [np.linspace(bounds[0][0], bounds[0][1],num_samples),np.linspace(bounds[1][0],bounds[1][1],num_samples)])  
	try:
		with open('plot_data.pkl', 'rb') as input:
			plot_data = pickle.load(input)
		plot_data = np.vstack((plot_data, [cont_rew, colorbar_rew, data_rew]))
	except:
		plot_data = [cont_rew, colorbar_rew, data_rew]
	# with open('plot_data.pkl', 'rb') as output:
	# 	pickle.dump(plot_data, output, pickle.HIGHEST_PROTOCOL)
	plt.show()
	# cont_constr, colorbar_constr, data_constr = my_safeopt.utilities.plot_contour_gp(gp_constr, [np.linspace(bounds[0][0], bounds[0][1],num_samples),np.linspace(bounds[1][0],bounds[1][1],num_samples)]) 
	cont_constr, colorbar_constr, data_constr = safeopt.utilities.plot_contour_gp(gp_constr, [np.linspace(bounds[0][0], bounds[0][1],num_samples),np.linspace(bounds[1][0],bounds[1][1],num_samples)]) 
	plot_data = np.vstack((plot_data, [cont_constr, colorbar_constr, data_constr]))
	with open('plot_data.pkl','wb') as output:
		pickle.dump(plot_data, output, pickle.HIGHEST_PROTOCOL)
	plt.show()

def furuta_safeopt(sim=True):
	safe = safe_opt(sim=sim)
	safe.setup_optimization()
	# for _ in range(1):
	safe.optimization()
	if not success:
		breakpoint()
	print("done!!!")

def try_furuta_real(param):
	frequency = 200
	divider = 4
	constr = np.inf
	with QubeSwingupEnv(use_simulator=True, frequency=frequency) as env:
		state = env.reset()
		swing_up_ctrl = QubeFlipUpControl(sample_freq=frequency, env=env)
		upright = False
		i = 0
		reward = 0
		while i < 1000:
			if upright:
				if np.abs(state[1]) < math.pi/2 and i % divider == 0:
					action = np.dot(param,state)
				elif np.abs(state[1]) >= math.pi/2:
					action = np.array([0.0])
					print("failed")
				state, rew, _, _ = env.step(action.flatten())
				reward += rew
				dist_constr = [np.pi/2 - np.abs(state[idx]) for idx in range(2)]
				if np.min(dist_constr)<constr:
					constr = np.min(dist_constr)
				i += 1
			else:
				action = swing_up_ctrl.action(state)
				state, _, _, _ = env.step(action)
				if np.linalg.norm(state) < 1e-2:
					print("swingup completed")
					upright = True 
		print(reward/1000)
		print(constr)

if __name__ == '__main__':
	warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
	try_furuta_real(np.array([-2.5712856, 23.011506, -1.5040040945983464, 3.0344775662414483]))
	furuta_safeopt()