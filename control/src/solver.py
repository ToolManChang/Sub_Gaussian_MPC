import timeit

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from acados_template import AcadosOcpSolver, AcadosSimSolver

from .utils.ocp import export_mpc_ocp  # , export_sim


# The class below is an optimizer class,
# it takes in GP function, x_g and rest are parameters
class MPC_solver(object):
    def __init__(self, params) -> None:
        ocp = export_mpc_ocp(params)
        self.name_prefix = (
            params["algo"]["type"]
            + "_env_"
            + str(params["env"]["name"])
            + "_i_"
            + str(params["env"]["i"])
            + "_"
        )
        self.ocp_solver = AcadosOcpSolver(
            ocp, json_file=self.name_prefix + "acados_ocp_sempc.json"
        )
        # self.ocp_solver.store_iterate(self.name_prefix + "ocp_initialization.json")

        # sim = export_sim(params, 'sim_sempc')
        # self.sim_solver = AcadosSimSolver(
        #     sim, json_file='acados_sim_sempc.json')
        self.H = params["optimizer"]["H"]
        self.Hm = params["optimizer"]["Hm"]
        self.max_sqp_iter = params["optimizer"]["SEMPC"]["max_sqp_iter"]
        self.tol_nlp = params["optimizer"]["SEMPC"]["tol_nlp"]
        self.nx = ocp.model.x.size()[0]
        self.nu = ocp.model.u.size()[0]
        self.n_order = params["optimizer"]["order"]
        self.x_dim = params["optimizer"]["x_dim"]
        self.u_dim = params["optimizer"]["u_dim"]
        self.state_dim = self.n_order * self.x_dim
        self.params = params

    def initilization(self, sqp_iter, x_h, u_h):
        for stage in range(1, self.H):
            # current stage values
            x_h[stage, :] = self.ocp_solver.get(stage + 1, "x")
            u_h[stage, :] = self.ocp_solver.get(stage, "u")
        x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
        if sqp_iter == 0:
            x_h_old = x_h.copy()
            u_h_old = u_h.copy()
            for stage in range(self.H):
                if stage < (self.H - self.Hm):
                    # current stage values
                    x_init = x_h_old[stage + self.Hm, :].copy()
                    u_init = u_h_old[stage + self.Hm, :].copy()
                    x_init[-1] = (
                        x_h_old[stage + self.Hm, -1] - x_h_old[self.Hm, -1]
                    ).copy()
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
                    half_time = x_init[-1].copy()
                else:
                    dt = (1.0 - half_time) / self.Hm
                    x_init = x_h_old[self.H, :].copy()  # reached the final state
                    x_init[-1] = half_time + dt * (stage - self.Hm)
                    z_init = x_init[0 : self.x_dim]
                    # u_init = np.array([0.0,0.0, dt]) # ?
                    u_init = np.zeros((self.u_dim))
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
            self.ocp_solver.set(self.H, "x", x_init)
            x_init[-1] = half_time + dt * (self.H - self.Hm)
            x_h[self.H, :] = x_init.copy()
            # x0 = np.zeros(self.state_dim)
            # x0[:self.x_dim] = np.ones(self.x_dim)*0.72
            # x0=np.concatenate([x0, np.array([0.0])])
            # x_init=x0.copy()
            # # x_init = self.ocp_solver.get(0, "x")
            # u_init = self.ocp_solver.get(0, "u")
            # Ts = 1/200
            # # MPC controller
            # x_init = np.array([0.72,0.72,0.0,0.0, 0.0])
            # u_init = np.array([-0.2,-0.2, Ts])

            #     x_h[stage, :] = x_init
            #     u_h[stage, :] = u_init
            # x_h[self.H, :] = x_init
            # self.ocp_solver.set(self.H, "x", x_init)
        return x_h, u_h

    def path_init(self, path):
        split_path = np.zeros((self.H + 1, self.x_dim))
        interp_h = np.arange(self.Hm)
        path_step = np.linspace(0, self.Hm, path.shape[0])
        x_pos = np.interp(interp_h, path_step, path.numpy()[:, 0])
        y_pos = np.interp(interp_h, path_step, path.numpy()[:, 1])
        split_path[: self.Hm, 0], split_path[: self.Hm, 1] = x_pos, y_pos
        split_path[self.Hm :, :] = (
            np.ones_like(split_path[self.Hm :, :]) * path[-1].numpy()
        )
        # split the path into horizons
        for stage in range(self.H + 1):
            x_init = self.ocp_solver.get(stage, "x")
            x_init[: self.x_dim] = split_path[stage]
            self.ocp_solver.set(stage, "x", x_init)

    def solve(self):
        x_h = np.zeros((self.H + 1, self.state_dim))
        u_h = np.zeros((self.H, self.x_dim))  # u_dim
        w = 1e-3 * np.ones(self.H + 1)
        we = 1e-8 * np.ones(self.H + 1)
        we[int(self.H - 1)] = 10000
        # w[:int(self.Hm)] = 1e-1*np.ones(self.Hm)
        w[int(self.Hm)] = self.params["optimizer"]["w"]
        xg = np.ones((self.H + 1, self.x_dim)) * np.array(
            self.params["env"]["goal_loc"]
        )
        # x_origin = player.origin[:self.x_dim].numpy()
        # x_terminal = np.zeros(self.state_dim)
        # x_terminal[:self.x_dim] = np.ones(self.x_dim)*x_origin
        for sqp_iter in range(self.max_sqp_iter):
            self.ocp_solver.options_set("rti_phase", 1)
            for stage in range(self.H):
                # current stage values
                x_h[stage, :] = self.ocp_solver.get(stage, "x")
                u_h[stage, :] = self.ocp_solver.get(stage, "u")
            x_h[self.H, :] = self.ocp_solver.get(self.H, "x")

            for stage in range(self.H + 1):
                self.ocp_solver.set(stage, "p", np.hstack((xg[stage], w[stage])))
            status = self.ocp_solver.solve()

            self.ocp_solver.options_set("rti_phase", 2)
            t_0 = timeit.default_timer()
            status = self.ocp_solver.solve()
            t_1 = timeit.default_timer()
            # self.ocp_solver.print_statistics()
            print("cost", self.ocp_solver.get_cost())
            residuals = self.ocp_solver.get_residuals()

            X, U, Sl = self.get_solution()
            # # print(X)
            # # for stage in range(self.H):
            # #     print(stage, " constraint ", self.constraint(LB_cz_val[stage], LB_cz_grad[stage], U[stage,3:5], X[stage,0:4], u_h[stage,-self.x_dim:], x_h[stage, :self.state_dim], self.params["common"]["Lc"]))
            # if sqp_iter==(self.max_sqp_iter-1):
            #     if self.params["visu"]["show"]:
            #         plt.figure(2)
            #         if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
            #             plt.plot(X[:,0],X[:,1], color="tab:green") # state
            #             plt.plot(U[:,3],U[:,4], color="tab:blue") # l(x)
            #         else:
            #             plt.plot(X[:,0],X[:,1], color="tab:green")
            #         plt.xlim(self.params["env"]["start"],self.params["env"]["start"] + self.params["visu"]["step_size"]*self.params["env"]["shape"]["x"])
            #         plt.ylim(self.params["env"]["start"],self.params["env"]["start"] + self.params["visu"]["step_size"]*self.params["env"]["shape"]["y"])
            #         # plt.axes().set_aspect('equal')
            #         plt.savefig("temp.png")
            # # print("statistics", self.ocp_solver.get_stats("statistics"))
            # if max(residuals) < self.tol_nlp:
            #     print("Residual less than tol", max(
            #         residuals), " ", self.tol_nlp)
            #     break
            # if self.ocp_solver.status != 0:
            #     print("acados returned status {} in closed loop solve".format(
            #         self.ocp_solver.status))
            #     self.ocp_solver.reset()
            #     self.ocp_solver.load_iterate(self.name_prefix + 'ocp_initialization.json')

    def solve_feedback(self, x_0, X_last, U_last, zeta=0.01):
        # variables
        x_h = np.zeros((self.H + 1, self.state_dim))
        u_h = np.zeros((self.H, self.u_dim))  # u_dim
        w = 1e-3 * np.ones(self.H + 1)
        we = 1e-8 * np.ones(self.H + 1)
        we[int(self.H - 1)] = 10000
        # w[:int(self.Hm)] = 1e-1*np.ones(self.Hm)
        w[int(self.Hm)] = self.params["optimizer"]["w"]
        xg = np.ones((self.H + 1, self.x_dim)) * np.array(
            self.params["env"]["goal_loc"]
        )

        # assign last solutions
        # x_h[:-1, :] = X_last[1, :]
        # u_h[:-1, :] = U_last[1, :]

        # assign the initial state
        x_h[0, :] = x_0

        for sqp_iter in range(self.max_sqp_iter):
            # initialization
            self.initilization(sqp_iter, x_h, u_h)
            # # print("optimizer x0", self.ocp_solver.get(0, "x"))
            # self.ocp_solver.options_set("rti_phase", 1)

            for stage in range(self.H + 1):
                self.ocp_solver.set(
                    stage, "p", np.hstack((xg[stage], w[stage]))
                )
            status = self.ocp_solver.solve()

            # self.ocp_solver.options_set("rti_phase", 2)
            # status = self.ocp_solver.solve()
            # self.ocp_solver.print_statistics()
            # print("cost", self.ocp_solver.get_cost())
            residuals = self.ocp_solver.get_residuals()

            x_h, u_h, Sl = self.get_solution()

        return x_h, u_h

    def constraint(self, lb_cz_lin, lb_cz_grad, model_z, model_x, z_lin, x_lin, Lc):
        x_dim = self.x_dim
        tol = 1e-5
        ret = (
            lb_cz_lin
            + lb_cz_grad.T @ (model_z - z_lin)
            - (Lc / (ca.norm_2(x_lin[:x_dim] - z_lin) + tol))
            * ((x_lin[:x_dim] - z_lin).T @ (model_x - x_lin)[:x_dim])
            - (Lc / (ca.norm_2(x_lin[:x_dim] - z_lin) + tol))
            * ((z_lin - x_lin[:x_dim]).T @ (model_z - z_lin))
            - Lc * ca.norm_2(x_lin[:x_dim] - z_lin)
        )
        # ret = lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - 2*Lc*(x_lin[:x_dim] - z_lin).T@(model_x-x_lin)[:x_dim] - 2*Lc*(z_lin-x_lin[:x_dim]).T@(model_z-z_lin) - Lc*(x_lin[:x_dim] - z_lin).T@(x_lin[:x_dim] - z_lin)
        return ret, lb_cz_lin + lb_cz_grad.T @ (model_z - z_lin)


    def get_solution(self):
        X = np.zeros((self.H + 1, self.nx))
        U = np.zeros((self.H, self.nu))
        Sl = np.zeros((self.H + 1))

        # get data
        for i in range(self.H):
            X[i, :] = self.ocp_solver.get(i, "x")
            U[i, :] = self.ocp_solver.get(i, "u")
            # Sl[i] = self.ocp_solver.get(i, "sl")

        X[self.H, :] = self.ocp_solver.get(self.H, "x")
        return X, U, Sl

    def get_solver_status():
        return None
