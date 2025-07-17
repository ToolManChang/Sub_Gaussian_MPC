import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Visualizer:
    def __init__(self, params, path):
        self.params = params
        self.nominal_pred_traj = []
        self.physical_traj = []

    def pendulum_discrete_dyn(self, X1_k, X2_k, U_k):
        """_summary_

        Args:
            x (_type_): _description_
            u (_type_): _description_
        """
        m = 1
        l = 1
        g = 10
        dt = self.params["optimizer"]["dt"]
        X1_kp1 = X1_k + X2_k * dt
        X2_kp1 = X2_k - g * np.sin(X1_k) * dt / l + U_k * dt / (l * l)
        return X1_kp1, X2_kp1

    def propagate_true_dynamics(self, x_init, U):
        x1_list = []
        x2_list = []
        X1_k = x_init[0]
        X2_k = x_init[1]
        x1_list.append(X1_k.item())
        x2_list.append(X2_k.item())
        for ele in range(U.shape[0]):
            X1_kp1, X2_kp1 = self.pendulum_discrete_dyn(X1_k, X2_k, U[ele])
            x1_list.append(X1_kp1.item())
            x2_list.append(X2_kp1.item())
            X1_k = X1_kp1.copy()
            X2_k = X2_kp1.copy()
        return x1_list, x2_list

    def propagate_mean_dyn(self, x_init, U):
        x1_list = []
        x2_list = []
        X1_k = x_init[0]
        X2_k = x_init[1]
        x1_list.append(X1_k.item())
        x2_list.append(X2_k.item())
        for ele in range(U.shape[0]):
            y1 = self.Dyn_gp_model["y1"](
                torch.Tensor([[X1_k, X2_k, U[ele]]])
            ).mean.detach()[0]
            y2 = self.Dyn_gp_model["y2"](
                torch.Tensor([[X1_k, X2_k, U[ele]]])
            ).mean.detach()[0]
            X1_kp1, X2_kp1 = y1[0], y2[0]
            x1_list.append(X1_kp1.item())
            x2_list.append(X2_kp1.item())
            X1_k = X1_kp1.clone()
            X2_k = X2_kp1.clone()
        return x1_list, x2_list

    def plot_traj(self):
        plt.close()
        traj = np.vstack(self.physical_traj)
        for i in range(len(self.nominal_pred_traj)):
            # plt.plot(self.nominal_pred_traj[i][:,0:2], '.' ,label = 'i')
            plt.plot(
                self.nominal_pred_traj[i][:, 0],
                self.nominal_pred_traj[i][:, 1],
                ".",
                label="i",
            )
        plt.plot(
            traj[:, 0], traj[:, 1], "k"
        )  # , label = [i for i in range(self.params["agent"]["num_dyn_samples"])])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.savefig("traj.png")

    def plot_traj_3D(self):
        plt.close()
        # Define range for x, y, and z
        z = np.linspace(-30, 30, 30)
        dz = self.params["env"]["funnel"]["dz"]
        dy = self.params["env"]["funnel"]["dy"]
        dx = self.params["env"]["funnel"]["dx"]
        k = 1
        y = np.linspace(-100, 100, 30)
        Z, Y = np.meshgrid(z, y)

        # Compute z values using the equation x^2 + y^2 = e^z
        zeta = self.params["env"]["funnel"]["zeta"]
        value = self.params["env"]["funnel"]["constraint"]
        X = (
            -k
            * dx
            * (
                np.log((Z / dz) ** 2 + (Y / dy) ** 2 - zeta - value)
                + self.params["env"]["funnel"]["shift"]
            )
        )

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # ax.plot_surface(X, Y, Z, cmap="viridis")
        ax.invert_zaxis()
        # ax.axis("equal")
        ax.contourf(Z, Y, X, cmap=cm.coolwarm)

        # Label axes
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")
        ax.set_zlabel("X")
        # ax.set_zlim(-1, 1)

        plt.title("Surface Plot of z^2 + y^2 -zeta - value = e^(x-shift)")
        traj = np.vstack(self.physical_traj)
        for i in range(len(self.nominal_pred_traj)):
            # plt.plot(self.nominal_pred_traj[i][:,0:2], '.' ,label = 'i')
            plt.plot(
                self.nominal_pred_traj[i][:, 2],
                self.nominal_pred_traj[i][:, 1],
                self.nominal_pred_traj[i][:, 0],
                ".",
                label="i",
            )
        plt.plot(
            traj[:, 2], traj[:, 1], traj[:, 0], "k"
        )  # , label = [i for i in range(self.params["agent"]["num_dyn_samples"])])
        # plt.xlabel("x")
        # plt.ylabel("y")
        plt.grid()
        plt.show()
        plt.savefig("traj.png")

    # def plot_traj(self, X, U):
    #     plt.close()
    #     plt.plot(X[:,0::2],X[:,1::2])#, label = [i for i in range(self.params["agent"]["num_dyn_samples"])])
    #     # plt.legend([i for i in range(self.params["agent"]["num_dyn_samples"])])
    #     plt.xlabel('theta')
    #     plt.ylabel('theta_dot')
    #     x1_true, x2_true = self.propagate_true_dynamics(X[0,0:2], U)
    #     plt.plot(x1_true, x2_true, color='black', label='true', linestyle='--')
    #     x1_mean, x2_mean = self.propagate_mean_dyn(X[0,0:2], U)
    #     print("x1_mean", x1_mean, x2_mean)
    #     plt.plot(x1_mean, x2_mean, color='black', label='mean', linestyle='-.')
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig('pendulum.png')
