import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from uncertainty.estimate_variance_proxy import *
from uncertainty.optimize_confidence_interval import *
from uncertainty.compute_confidence_bound import *
from uncertainty.variance_proxy_propagation import *
from envs import *
from controllers import *
from state_estimator import *
from visualize import *
from ruamel.yaml import YAML
from numerical import *
from tueplots import bundles
from tueplots.constants.color import palettes
from tueplots import figsizes


def test_trajectory_mpc(
    if_sub_gau: bool,
    mean_x_0: np.ndarray,
    goal: np.ndarray,
    env: LinearEnv,
    cfg: dict,
    prob: float,
    num_steps: int,
    num_samples: int,
    ax: matplotlib.axes.Axes,
):
    """
    env: linear env
    policy: any policy
    prob: small prob of outside the bound
    num_steps: number of steps
    num_samples: number of sampling the trajectory
    """

    # get noise samples
    dist_cfg = env.noise_cfg["disturbance"]
    meas_cfg = env.noise_cfg["measurement"]
    dists = []
    meas = []
    init_states = []
    for i in range(num_samples * 5):
        dists.append(env.sample_noise(dist_cfg, env.A.shape[0]))
        meas.append(env.sample_noise(meas_cfg, env.C.shape[0]))
        init_states.append(
            env.sample_noise(env.noise_cfg["init_state"], env.A.shape[0])
        )
    dists = np.asarray(dists)
    meas = np.asarray(meas)
    init_states = np.asarray(init_states)

    if "state_variant" in env.noise_cfg:
        dists_2 = []
        meas_2 = []
        init_states_2 = []
        scale = env.noise_cfg["state_variant"]["above_scale"]
        for i in range(num_samples * 2):
            dists_2.append(env.sample_noise(dist_cfg, env.A.shape[0]) * scale)
            meas_2.append(env.sample_noise(meas_cfg, env.C.shape[0]) * scale)
            init_states_2.append(
                env.sample_noise(env.noise_cfg["init_state"], env.A.shape[0]) * scale
            )
        dists_2 = np.asarray(dists_2)
        meas_2 = np.asarray(meas_2)
        init_states_2 = np.asarray(init_states_2)

        dists = np.concatenate((dists, dists_2), axis=0)
        meas = np.concatenate((meas, meas_2), axis=0)
        init_states = np.concatenate((init_states, init_states_2), axis=0)

    # estimate variance proxy
    if if_sub_gau:
        if "state_variant" in env.noise_cfg:
            vp_w, _ = estimate_variance_proxy(dists_2)
            vp_e, _ = estimate_variance_proxy(meas_2)
            vp_0, _ = estimate_variance_proxy(init_states_2)
        else:
            vp_w, _ = estimate_variance_proxy(dists)
            vp_e, _ = estimate_variance_proxy(meas)
            vp_0, _ = estimate_variance_proxy(init_states)

        print("vp of dist: ", vp_w)
        print("vp of meas:", vp_e)
        print("vp of init:", vp_0)

        vp_w = vp_w**2 * np.eye(env.A.shape[0])
        vp_e = vp_e**2 * np.eye(env.C.shape[0])
        vp_0 = vp_0**2 * np.eye(env.A.shape[0])
    else:
        vp_w = dists.T @ dists / 5 / num_samples
        vp_e = meas.T @ meas / 5 / num_samples
        vp_0 = init_states.T @ init_states / 5 / num_samples

    # rollout kalman gain
    mpc_controller = MPCController(cfg)
    mpc_controller.set_env(env)

    # if robust
    if mpc_controller.mpc_params["approach"] == "robust":
        poly_w = np.max(np.linalg.norm(dists, axis=1))
        poly_e = np.max(np.linalg.norm(meas, axis=1))
        poly_0 = np.max(np.linalg.norm(init_states, axis=1))

        # get squared norm for propagation
        poly_w = poly_w**2 * np.eye(env.A.shape[0])
        poly_e = poly_e**2 * np.eye(env.C.shape[0])
        poly_0 = poly_0**2 * np.eye(env.A.shape[0])

    # apply control to sample trajectories
    all_zs = []
    all_xs = []
    all_x_ests = []
    all_us = []
    for sample in range(int(num_samples / 10)):
        # reset the env
        x, y = env.reset(mean_x_0)

        if sample == 0 or mpc_controller.mpc_params["approach"] == "nominal":
            if mpc_controller.mpc_params["approach"] == "robust":
                u, v = mpc_controller.init_mpc(
                    y,
                    mean_x_0,
                    goal,
                    vp_0,
                    vp_w,
                    vp_e,
                    None,
                    poly_0,
                    poly_w,
                    poly_e,
                    num_steps,
                    prob,
                )
            else:
                u, v = mpc_controller.init_mpc(
                    y,
                    mean_x_0,
                    goal,
                    vp_0,
                    vp_w,
                    vp_e,
                    v=None,
                    num_steps=num_steps,
                    prob=prob,
                )
        else:
            if mpc_controller.mpc_params["approach"] == "robust":
                u, v = mpc_controller.init_mpc(
                    y,
                    mean_x_0,
                    goal,
                    vp_0,
                    vp_w,
                    vp_e,
                    vs[0],
                    poly_0,
                    poly_w,
                    poly_e,
                    num_steps,
                    prob,
                )
            else:
                u, v = mpc_controller.init_mpc(
                    y,
                    mean_x_0,
                    goal,
                    vp_0,
                    vp_w,
                    vp_e,
                    vs[0],
                    num_steps=num_steps,
                    prob=prob,
                )

        zs = [mpc_controller.z]
        xs = [x]
        x_ests = [mpc_controller.x_est]
        us = [u]
        if sample == 0:
            vs = [v]
        else:
            print("approach:", mpc_controller.mpc_params["approach"])
            print("sample:", sample)
        for t in range(num_steps):

            x, y = env.step(x, u)

            if mpc_controller.mpc_params["approach"] == "nominal" or sample == 0:
                u, v = mpc_controller.mpc_policy(y, t + 1)
                vs.append(v)
            else:
                v = vs[t + 1]
                u = mpc_controller.mpc_policy_given_v(y, t + 1, v)

            zs.append(mpc_controller.z)
            xs.append(x)
            x_ests.append(mpc_controller.x_est)
            us.append(u)
            if sample == 0:
                print("t:", t)
                print("z: ", mpc_controller.z)
                print("x: ", x)

        all_zs.append(zs)
        all_xs.append(xs)
        all_x_ests.append(x_ests)
        all_us.append(us)

    if (
        mpc_controller.mpc_params["approach"] == "sub-Gaussian"
        and mpc_controller.mpc_params["tube"]["shape"] == "ellipsoid"
    ):
        color = palettes.tue_plot[4]  # green
    elif (
        mpc_controller.mpc_params["approach"] == "sub-Gaussian"
        and mpc_controller.mpc_params["tube"]["shape"] == "half-space"
    ):
        color = palettes.tue_plot[3]  # blue
    elif mpc_controller.mpc_params["approach"] == "Gaussian":
        color = palettes.tue_plot[7]  # pink
    else:
        color = palettes.tue_plot[5]
    draw_1d_trajectories(
        ax,
        all_xs,
        all_zs,
        all_x_ests,
        env.cfg["interest_dim"],
        mpc_controller,
        color=color,
        alpha=0.3,
    )

    unsafe_ratio = compute_average_fail_number(all_xs, env) / num_steps
    max_unsafe_ratio = compute_max_fail_number(all_xs, env) / num_steps

    average_cost = compute_average_cost(
        all_xs,
        all_us,
        mpc_controller.Q,
        mpc_controller.R,
        np.array(mpc_controller.mpc_params["env"]["goal_loc"]),
    )

    return max_unsafe_ratio, unsafe_ratio, average_cost


if __name__ == "__main__":
    bundle = bundles.icml2024()
    bundle["legend.fontsize"] = 11
    bundle["font.size"] = 11
    plt.rcParams.update(bundle)
    plt.rcParams.update(figsizes.icml2024_half())
    plt.rcParams["figure.figsize"] = [6.0, 3.5]
    plt.rcParams.update(
        {
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    # set the random seed
    unsafe_ratio = 0.05
    cfg = YAML().load(open("cfgs/envs/mass_spring_damper.yaml"))
    integrator_env = LinearEnv(cfg)
    # test trajectory propagation
    fig, axes = plt.subplots(1, 1)
    # mpc
    # set the initial state
    mean_x_0 = np.array(cfg["start_loc"])
    goal = np.array(cfg["goal_loc"])
    mpc_cfg = YAML().load(open("cfgs/controller/tube_ellipsoid_params.yaml"))
    mpc_cfg["approach"] = "sub-Gaussian"
    mpc_cfg["tube"]["shape"] = "half-space"
    np.random.seed(0)

    fig2, axes2 = plt.subplots(1, 2)
    np.random.seed(0)
    maxout_subGau, out_subGau, cost_subGau = test_trajectory_mpc(
        True, mean_x_0, goal, integrator_env, mpc_cfg, unsafe_ratio, 100, 1000, axes2[0]
    )

    mpc_cfg["approach"] = "DR"
    np.random.seed(0)
    maxout_DR, out_DR, cost_DR = test_trajectory_mpc(
        False,
        mean_x_0,
        goal,
        integrator_env,
        mpc_cfg,
        unsafe_ratio,
        100,
        1000,
        axes2[1],
    )

    mpc_cfg["approach"] = "nominal"
    axes2[0].set_title("Sub-Gaussian SMPC (Half-space)")
    axes2[0].set_xlabel("Time")
    axes2[0].set_ylabel("x[0]")
    axes2[0].set_ylim(-0.2, 1.75)

    axes2[1].set_title("DR SMPC")
    axes2[1].set_xlabel("Time")
    axes2[1].set_ylabel("x[0]")
    axes2[1].set_ylim(-0.2, 1.75)

    axes2[0].legend(loc="upper right")
    axes2[1].legend(loc="upper right")
    fig2.savefig(
        "images/mpc_classic_" + str(integrator_env.cfg["name"]) + "_all.pdf", dpi=300
    )
    print("Sub-Gaussian out ratio:", out_subGau)
    print("DR out ratio:", out_DR)

    print("Sub-Gaussian cost:", cost_subGau)
    print("DR cost:", cost_DR)

    print("Sub-Gaussian max out ratio:", maxout_subGau)
    print("DR max out ratio:", maxout_DR)
