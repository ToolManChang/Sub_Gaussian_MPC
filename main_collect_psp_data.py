'''
collect data to estimate the dynamic and observation uncertainty of PSP
'''


import gym
import numpy as np
from ruamel.yaml import YAML


num_env = 1
env_id = "PedicleScrewPlacement:simple-v0"
cfg_file = 'cfgs/envs/psp.yaml'
env = gym.make(env_id, cfg_file=cfg_file)
cfg = YAML().load(open(cfg_file))
env.cfg['reset']['x_range'] = [-10, -5]
env.cfg['reset']['y_range'] = [-20, -15]
env.cfg['reset']['z_angle_range'] = [0.3, 0.4]
test_num = 20
max_step = 200
start_step = 30

est_states = []
gt_states = []
est_errs = []
dyn_errs = []

for test in range(test_num):

    obs = env.reset()
    pred_state = env.control_state

    for step in range(max_step):

        action = [0.0, 0.0, 0.0, 0.0, 0.0]

        obs, rewards, dones, info = env.step(action)

        est_states.append(env.control_state)
        gt_states.append(env.gt_control_state)
        

        if step > start_step:
            err = env.control_state - env.gt_control_state
            dyn_err = env.control_state - pred_state
            est_errs.append(err)
            dyn_errs.append(dyn_err)
            print('est_err:', err)
            print('dyn_err:', dyn_err)

        pred_state = env.control_state

    est_err_array = np.array(est_errs)
    dyn_err_array = np.array(dyn_errs)
    np.save('data/psp/est_errs.npy', est_err_array)
    np.save('data/psp/dyn_errs.npy', dyn_err_array)

        