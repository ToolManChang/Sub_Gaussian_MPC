import numpy as np
from state_estimator import kalman_filter_gain_rollout
from uncertainty.estimate_variance_proxy import *
from uncertainty.optimize_confidence_interval import *
from uncertainty.compute_confidence_bound import *
from uncertainty.variance_proxy_propagation import *
# from .registration import pcd_to_3d_image_coords
# The environment for this file is a ~/work/MPC_RL_SPINE
import os

from ruamel.yaml import YAML

from control.src.MPC import MPC
from scipy.spatial.transform import Rotation as R
import numpy as np
from envs import *
from state_estimator import *
import scipy
import pyvista as pv
from visualize import get_ellipsoids


# robust

# Indirect

# direct

# nominal


def solve_LQR(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    K = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    return P, K


class MPCController:

    def __init__(self, cfg) -> None:
        '''
        init a MPC controller and a state estimater
        '''
        # TODO Tube MPC contoller (from MPC class)
        # 1) Load the config file
        self.mpc_params = cfg
        
        pass


    def set_env(self, env: LinearEnv):
        '''
        env: linear env
        '''
        self.mpc_params["env"] = env.cfg
        self.mpc_params["env"]["i"] = 226
        # self.mpc_params["env"]["name"] = 2
        self.mpc_params["env"]["dummy"] = 1
        # self.mpc_params["env"]["A"] = env.A
        # self.mpc_params["env"]["B"] = env.B
        # self.mpc_params["env"]["C"] = env.C
        # self.mpc_params["env"]["Q"] = env.Q
        # self.mpc_params["env"]["R"] = env.R
        self.mpc_params["optimizer"]["u_max"] = env.u_max
        self.mpc_params["optimizer"]["u_min"] = env.u_min
        self.mpc_params["optimizer"]["x_max"] = env.x_max
        self.mpc_params["optimizer"]["x_min"] = env.x_min
        self.mpc_params["optimizer"]["x_dim"] = env.A.shape[0]
        self.mpc_params["optimizer"]["u_dim"] = env.B.shape[1]
        self.A = env.A
        self.B = env.B
        self.C = env.C
        self.Q = env.Q
        self.R = env.R
        self.dt = env.dt
        self.mpc_params["env"]["constraints"] = env.constraint_cfg
        self.mpc_params['optimizer']['dt'] = env.dt
        self.cons_dim = env.cfg['constraints']['num_dim']
        

        pass


    def init_mpc(self, y0, start, goal, vp_0, vp_w, vp_e, v=None, poly_0=None, poly_w=None, poly_e=None, num_steps=100, prob=0.01):
        '''
        y0: initial observation
        start: mean of initial state
        goal: goal state
        vp_0: variance proxy or polygon of initial state
        vp_w: variance proxy or polygon of disturbance 
        vp_e: variance proxy or polygon of measurement noise
        v: init nominal action
        poly_0: polygon / robust ellipsoid of initial state
        poly_w: polygon / robust ellipsoid of disturbance
        poly_e: polygon / robust ellipsoid of measurement noise
        prob: small prob of outside the bound
        '''

        # init MPC
        self.K = self.mpc_params["tube"]["K"]
        self.mpc_params["env"]["start_loc"] = start
        self.mpc_params["env"]["goal_loc"] = goal


        # solve LQR
        self.solve_LQR()

        # assign noise uncertainties
        self.vp_0 = vp_0
        self.vp_w = vp_w
        self.vp_e = vp_e

        # poly uncertainty
        if self.mpc_params['approach'] == 'robust':
            self.poly_0 = poly_0
            self.poly_w = poly_w
            self.poly_e = poly_e

        # propagate uncertainties
        self.offline_kalman_gain_propagation(num_steps)
        self.offline_uncertainty_propagation(num_steps)

        # constraints tightening
        if self.mpc_params['approach'] == 'robust':
            self.tightens = self.constraints_tightening_robust()
        else:
            self.tightens = self.constraints_tightening_prob(prob)
            
        self.mpc_params["tube"]["tightening"] = self.tightens[-1]

        # init mpc
        if v is None or self.mpc_params['approach'] == 'nominal':
            self.mpc = MPC(self.mpc_params, None)

        # init KF
        self.z = start
        self.x_est = self.z + self.Ls[0] @ (y0 - self.C @ self.z)
        self.s_x = vp_0 # variance proxy of the state

        # init action
        if self.mpc_params['approach'] == 'nominal':
            self.X, U = self.mpc.receding_horizon(self.x_est)
            self.u = U
            self.v = np.zeros_like(U)

        elif v is None:
            self.X, U = self.mpc.receding_horizon(self.z)

            self.v = U

            self.u = self.K @ (self.x_est - self.z) + self.v
        else:
            self.v = v
            self.u = self.K @ (self.x_est - self.z) + v

        return self.u, self.v
    

    def solve_LQR(self):
        
        # solve riccati equation
        P, K = solve_LQR(self.A, self.B, self.Q, self.R)

        self.mpc_params["tube"]["K"] = K
        self.mpc_params["env"]["P"] = P.tolist()
        self.K = K


    def offline_kalman_gain_propagation(self, num_steps):
    
        self.Ls = kalman_filter_gain_rollout(
            self.vp_0, 
            self.A,
            self.C,
            self.vp_w, 
            self.vp_e, 
            num_steps
        )

        pass


    def offline_uncertainty_propagation(self, num_steps):
        if not self.mpc_params['approach'] == 'robust':
            self.vp_ests, self.vp_tracks = vp_propagation_est_track(
                self.vp_0,
                self.vp_w,
                self.vp_e,
                self.A,
                self.B,
                self.C,
                self.K,
                self.Ls,
                num_steps
            )
        elif self.mpc_params['approach'] == 'robust':
            if self.mpc_params['tube']['robust_approach']=='polygon':
                self.poly_ests, self.poly_track_ests, self.poly_track_trues = robust_propagation(
                    self.poly_0,
                    self.poly_w,
                    self.poly_e,
                    self.A,
                    self.B,
                    self.C,
                    self.K,
                    self.Ls,
                    num_steps)
            else:
                # here robust propagation is same as the sub gaussion norm propagation
                self.poly_ests, self.poly_tracks = robust_ellipsoid_propagation(
                    self.poly_0,
                    self.poly_w,
                    self.poly_e,
                    self.A,
                    self.B,
                    self.C,
                    self.K,
                    self.Ls,
                    num_steps
                )
        pass


    def constraints_tightening_prob(self, prob):
        # constraints tightening
        m = optimize_confidence_interval(prob, self.cons_dim)[0]

        # bound
        if self.mpc_params['approach'] == 'sub-Gaussian':
            if len(self.mpc_params["env"]["constraints"]['a'])==1 and self.mpc_params['tube']['shape']=='half-space':
                scale = np.sqrt(2 * np.log(1 / prob)) # use the half space bound
            else: # a larger bound
                scale = get_bound_scale_given_probability_from_m(prob, m, self.cons_dim)
        elif self.mpc_params['approach'] == 'Gaussian': 
            if len(self.mpc_params["env"]["constraints"]['a'])==1 and not self.mpc_params["env"]["constraints"]['funnel']['if_funnel']:
                scale = gaussian_scale_from_prob(prob, 1, [1e-7, 10], 10000)
            else:
                scale = gaussian_scale_from_prob(prob, self.cons_dim, [1e-7, 10], 10000)
        elif self.mpc_params['approach'] == 'nominal':
            scale = 0
        elif self.mpc_params['approach'] == 'DR':
            scale = np.sqrt(self.cons_dim / prob)

        tightens = []
        self.track_err_bounds = []

        # tighten the constraints
        for vp in self.vp_tracks:
            vp_true_to_nominal = vp[-self.A.shape[0]:, -self.A.shape[0]:]
            eigvals, eigvecs = np.linalg.eig(vp_true_to_nominal)
            tightens.append({})
            self.track_err_bounds.append(scale**2 * vp_true_to_nominal)

            if self.mpc_params["env"]["constraints"]['circle']:
                tightens[-1]["circle"]=scale * max(np.sqrt(eigvals))
            if self.mpc_params["env"]["constraints"]['exponential']:
                tightens[-1]["exponential"]=scale * max(np.sqrt(eigvals))
            if self.mpc_params["env"]["constraints"]['polygon']:
                a = np.array(self.mpc_params["env"]["constraints"]['a'])
                margin = scale * np.sqrt(np.diagonal(a @ vp_true_to_nominal @ a.T))
                tightens[-1]['polygon']=margin
            if self.mpc_params["env"]["constraints"]['funnel']['if_funnel']:
                a = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
                margin = np.sqrt(a @ vp_true_to_nominal @ a.T)
                b = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
                margin = max(margin, np.sqrt(b @ vp_true_to_nominal @ b.T))
                tightens[-1]['funnel']=margin
            
        return tightens

    def constraints_tightening_robust(self):

        tightens = []
        self.track_err_bounds = []
    
        if self.mpc_params['tube']['robust_approach']=='polygon':
            for poly in self.poly_track_trues:
                points = poly
                tightens.append({})
                if self.mpc_params["env"]["constraints"]['circle']:
                    tightens[-1]["circle"]=np.max(np.linalg.norm(points, axis=1))
                if self.mpc_params["env"]["constraints"]['exponential']:
                    tightens[-1]["exponential"]=np.max(np.linalg.norm(points, axis=1))
                if self.mpc_params["env"]["constraints"]['polygon']:
                    a = np.array(self.mpc_params["env"]["constraints"]['a'])
                    margin = np.max(np.abs(a @ points.T))
                    tightens[-1]["polygon"]=margin
        else:
            for vp in self.poly_tracks:
                vp_true_to_nominal = vp[-self.A.shape[0]:, -self.A.shape[0]:]
                eigvals, eigvecs = np.linalg.eig(vp_true_to_nominal)
                self.track_err_bounds.append(vp_true_to_nominal)
                tightens.append({})
                if self.mpc_params["env"]["constraints"]['circle']:
                    tightens[-1]['circle']=np.max(np.sqrt(eigvals))
                if self.mpc_params["env"]["constraints"]['exponential']:
                    tightens[-1]['exponential']=np.max(np.sqrt(eigvals))
                if self.mpc_params["env"]["constraints"]['funnel']['if_funnel']:
                    a = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
                    margin_a = np.sqrt(a @ vp_true_to_nominal @ a.T)
                    b = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
                    margin_b = np.sqrt(b @ vp_true_to_nominal @ b.T)
                    c = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
                    margin_c = np.sqrt(c @ vp_true_to_nominal @ c.T)
                    margin = np.sqrt(margin_a**2 + margin_b**2 + margin_c**2)
                    tightens[-1]['funnel']=margin
                if self.mpc_params["env"]["constraints"]['polygon']:
                    a = np.array(self.mpc_params["env"]["constraints"]['a'])
                    margin = np.sqrt(np.diagonal(a @ vp_true_to_nominal @ a.T))
                    tightens[-1]['polygon']=margin

        return tightens


    def mpc_policy(self, y, t):
        
        self.z = self.A @ self.z + self.B @ self.v

        self.x_est, self.s_x = kalman_filter_fix_gain(
            self.x_est,
            self.u,
            self.s_x,
            y,
            self.A,
            self.B,
            self.C,
            self.Ls[t],
            self.vp_w,
            self.vp_e
        )

        if self.mpc_params['approach'] == 'nominal':
            self.X, U = self.mpc.receding_horizon(self.x_est)
            self.u = U
            self.v = self.u

        else:
            self.X, U = self.mpc.receding_horizon(self.z)

            self.v = U

            self.u = self.K @ (self.x_est - self.z) + self.v

        return self.u, self.v
    

    def mpc_policy_given_v(self, y, t, v):
        '''
        fast computation of policy without solving v
        should not be used for nominal MPC
        '''
        
        self.z = self.A @ self.z + self.B @ self.v

        self.x_est, self.s_x = kalman_filter_fix_gain(
            self.x_est,
            self.u,
            self.s_x,
            y,
            self.A,
            self.B,
            self.C,
            self.Ls[t],
            self.vp_w,
            self.vp_e
        )

        self.v = v

        self.u = self.K @ (self.x_est - self.z) + v

        return self.u
    


    def init_ellipsoids(self, interval):
        self.ellipsoids = get_ellipsoids(self.track_err_bounds, interval)
    

    def rotate_ellipsoids(self, control_origin, right_traj_rot):
        # only need once
        self.human_ellipsoids = []
        # convert to the human frame
        for i in range(len(self.ellipsoids)):
            ellipsoid_pos = self.ellipsoids[i].points * 1000
            human_ellipsoid_pos = ellipsoid_pos @ right_traj_rot.T + control_origin
            self.ellipsoids[i].points = human_ellipsoid_pos
            human_ellipsoid = pv.PolyData(human_ellipsoid_pos)
            # move outside (hide)
            human_ellipsoid.points -= 1000
            self.human_ellipsoids.append(human_ellipsoid)

    
    def rotate_trajectory(self, control_origin, right_traj_rot):
        '''
        rotate the trajectory to the human frame
        '''
        # convert to the human frame
        traj_pos =self.X[:, :3] * 1000
        self.X_human = traj_pos @ right_traj_rot.T + control_origin
        self.X_human = self.X_human
    

    def update_ellipsoids(self, cur_step):
        '''
        move ellipsoid based on current pred trajectories
        '''
        for i in range(self.X_human.shape[0]):
            index = min(len(self.track_err_bounds)-1, i + cur_step)
            self.human_ellipsoids[index].points += (
                self.X_human[i, :] 
                - self.human_ellipsoids[index].center)





class DummyController:

    def __init__(self, K, L, v_seq) -> None:

        self.K = K
        self.L = L
        self.v_seq = v_seq
        self.i = 0
        
        pass

    def policy(self, x_est, z):
        '''
        x_est: est state
        z: nominal state
        '''
        v = self.v_seq[self.i % len(self.v_seq)]
        u = v + self.K @ (x_est - z)
        self.i += 1

        return u, v