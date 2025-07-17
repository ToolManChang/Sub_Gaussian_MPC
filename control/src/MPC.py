import os

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from .solver import MPC_solver

# from src.utils.helper import (TrainAndUpdateConstraint, TrainAndUpdateDensity,
#                               get_frame_writer, oracle)
# from src.utils.initializer import get_players_initialized
# from src.utils.termcolor import bcolors


class MPC:
    def __init__(self, params, visu) -> None:
        self.mpc_solver = MPC_solver(params)
        self.params = params
        self.iter = -1
        self.data = {}
        self.flag_reached_goal = False
        self.flag_new_goal = True

        self.prev_goal_dist = 100
        self.goal_in_pessi = False
        self.visu = visu
        # if params["agent"]["dynamics"]=="robot":
        #     self.state_dim = self.n_order*self.x_dim + 1
        # else:
        #     self.state_dim = self.n_order*self.x_dim
        # self.mpc_initialization()
        self.X = np.ones(
            (self.params["optimizer"]["H"] + 1, self.params["optimizer"]["x_dim"])
        ) * np.array(self.params["env"]["start_loc"])
        self.U = np.zeros(
            (self.params["optimizer"]["H"], self.params["optimizer"]["u_dim"])
        )

    def main(self):
        X = np.ones(
            (self.params["optimizer"]["H"], self.params["optimizer"]["x_dim"])
        ) * np.array(self.params["env"]["start_loc"])
        self.visu.physical_traj.append(X[0])
        x_curr_nominal = X[0]
        x_curr = X[0]
        K = -2
        i = 0
        while not self.flag_reached_goal and i < 1000:
            i += 1
            X, U = self.receding_horizon(x_curr_nominal)
            self.visu.nominal_pred_traj.append(X)
            u_apply = K * (x_curr - x_curr_nominal) + U
            x_curr = self.simulator_location_euler(x_curr, u_apply)
            self.visu.physical_traj.append(x_curr)
            x_curr_nominal = X[1]
            # self.simulator_no_noise(x_curr_nominal, u_apply) #X[1]
            error = np.linalg.norm((X[0] - self.params["env"]["goal_loc"])[0:3], 2)
            print(X[0], U[0], error)
            if error < 10e1:
                self.flag_reached_goal = True
                print("Goal reached")
                self.visu.plot_traj_3D()
                break


    def receding_horizon(self, x_0):
        # initialize the x_0
        self.mpc_solver.ocp_solver.set(0, "lbx", x_0)
        self.mpc_solver.ocp_solver.set(0, "ubx", x_0)
        self.mpc_solver.solve_feedback(x_0, self.X, self.U)
        self.X, self.U, Sl = self.mpc_solver.get_solution()
        return self.X, self.U[0]
