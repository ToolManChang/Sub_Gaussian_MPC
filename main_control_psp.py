import jax
import jax.nn as jnn
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from jaxopt._src.lbfgs import LbfgsState
from scipy import stats
import scipy.stats as st
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
import gym
from typing import Any
import copy
from tueplots import axes, figsizes


def test_trajectory_mpc(
    approach: str,
    mean_x_0: np.ndarray,
    goal: np.ndarray,
    env: Any,
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
    dists = np.load("data/psp/0930/dyn_errs.npy") / 1000
    meas = np.load("data/psp/0930/est_errs.npy") / 1000
    init_states = np.random.uniform(-0.0005, 0.0005, (num_samples, 5))

    # estimate variance proxy
    if approach == "sub-Gaussian":
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
        dist_mean = np.mean(dists, axis=0)
        meas_mean = np.mean(meas, axis=0)
        init_mean = np.mean(init_states, axis=0)
        # vp_w = (dists - dist_mean).T @ (dists - dist_mean) / dists.shape[0]
        # vp_e = (meas - meas_mean).T @ (meas - meas_mean) / meas.shape[0]
        # vp_0 = (init_states - init_mean).T @ (init_states - init_mean) / init_states.shape[0]
        vp_w = np.diag(np.var(dists, axis=0))
        vp_e = np.diag(np.var(meas, axis=0))
        vp_0 = np.diag(np.var(init_states, axis=0))

    # compute confidence interval
    if approach == "sub-Gaussian":
        m = optimize_confidence_interval(prob, env.cfg["constraints"]["num_dim"])[0]
        print("m:", m)  # from varaice proxy to bound

        # get bound scale
        # gaussian and sub ga
        scale = get_bound_scale_given_probability_from_m(
            prob, m, env.cfg["constraints"]["num_dim"]
        )
        cfg["approach"] = "sub-Gaussian"
    elif approach == "Gaussian":
        scale = gaussian_scale_from_prob(
            prob, env.cfg["constraints"]["num_dim"], [1e-7, 10], 10000
        )
        cfg["approach"] = "Gaussian"
        print("scale", scale)
    elif approach == "DR":
        scale = np.sqrt(env.cfg["constraints"]["num_dim"] / prob)
        cfg["approach"] = "DR"

    # rollout kalman gain
    mpc_controller = MPCController(cfg)
    mpc_controller.set_env(env)

    # if robust
    if approach == "robust":
        mpc_controller.mpc_params["approach"] = "robust"
        if mpc_controller.mpc_params["tube"]["robust_approach"] == "ellipsoid":
            poly_w = np.max(np.linalg.norm(dists, axis=1))
            poly_e = np.max(np.linalg.norm(meas, axis=1))
            poly_0 = np.max(np.linalg.norm(init_states, axis=1))

            # get squared norm for propagation
            poly_w = poly_w**2 * np.eye(env.A.shape[0])
            poly_e = poly_e**2 * np.eye(env.C.shape[0])
            poly_0 = poly_0**2 * np.eye(env.A.shape[0])

            scale = 1

        else:
            poly_w = compute_bound_points(dists)
            poly_e = compute_bound_points(meas)
            poly_0 = compute_bound_points(init_states)

            poly_w = np.max(np.linalg.norm(dists, axis=1))
            poly_e = np.max(np.linalg.norm(meas, axis=1))
            poly_0 = np.max(np.linalg.norm(init_states, axis=1))

            # get squared norm for propagation
            poly_w = poly_w**2 * np.eye(env.A.shape[0])
            poly_e = poly_e**2 * np.eye(env.C.shape[0])
            poly_0 = poly_0**2 * np.eye(env.A.shape[0])

    if approach == "nominal":
        mpc_controller.mpc_params["approach"] = "nominal"

    # apply control to sample trajectories
    all_zs = []
    all_xs = []
    all_x_ests = []
    all_us = []
    vs = []

    obs = env.reset()

    # init mean_x_0
    mean_x_0 = env.screw_traj_poses_to_control_state(
        screw_center=mean_x_0[0:3],
        screw_rot=np.eye(3),
        control_origin=copy.deepcopy(env.control_origin) / 1000,
        control_rot=env.right_traj_rot.as_matrix(),
    )

    for sample in range(int(num_samples / 10)):

        # reset the env
        obs = env.reset()
        goal = np.asarray([env.right_traj_len / 2000 + 0.09, 0.0, 0.0, 0.0, 0.0])
        # mean_x_0 = copy.deepcopy(env.control_state)
        # mean_x_0[0:3] /= 1000

        org_bound = env.cfg["motion_noise"]["bound"]
        env.cfg["motion_noise"]["bound"] = 0
        for step in range(int(num_steps / 5)):
            print("control_state", env.control_state)
            obs, rewards, dones, info = env.step(np.zeros((5,)).tolist())
        env.cfg["motion_noise"]["bound"] = org_bound

        x = copy.deepcopy(env.gt_control_state)
        y = copy.deepcopy(env.control_state)
        x[0:3] /= 1000
        y[0:3] /= 1000

        if sample == 0 or approach == "nominal":
            if approach == "robust":
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
            if approach == "robust":
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
            print("approach:", approach)
            print("sample:", sample)

        # visualize
        if env.cfg["visualize_full"]:
            mpc_controller.init_ellipsoids(1)
            mpc_controller.rotate_ellipsoids(
                env.control_origin, env.right_traj_rot.as_matrix()
            )
            for ellipsoid in mpc_controller.human_ellipsoids:
                env.p.add_mesh(ellipsoid, color=[204, 204, 255], opacity=0.0005)

        for t in range(num_steps):

            # apply action to env
            apply_u = copy.deepcopy(np.asarray(u))
            # convert to mm
            apply_u[0:3] *= 1000
            apply_u *= env.dt
            obs, rewards, dones, info = env.step(apply_u.tolist())

            x = copy.deepcopy(env.gt_control_state)
            y = copy.deepcopy(env.control_state)
            x[0:3] /= 1000
            y[0:3] /= 1000

            if approach == "nominal" or sample == 0:
                u, v = mpc_controller.mpc_policy(y, t + 1)
                vs.append(v)
            else:
                v = vs[t + 1]
                u = mpc_controller.mpc_policy_given_v(y, t + 1, v)

            zs.append(mpc_controller.z)
            xs.append(x)
            x_ests.append(mpc_controller.x_est)
            us.append(u)

            if env.cfg["visualize_full"]:
                mpc_controller.rotate_trajectory(
                    env.control_origin, env.right_traj_rot.as_matrix()
                )
                mpc_controller.update_ellipsoids(t + 1)

            if sample == 0:
                print("approach:", approach)
                print("sample:", sample)
                print("t:", t)
                print("z:", mpc_controller.z)
                print("x:", x)
                print("x_est", mpc_controller.x_est)
                print("v:", v)
                print("u:", u)
                if sample > 0:
                    print("original z:", all_zs[0][t + 1])

        all_zs.append(zs)
        all_xs.append(xs)
        all_x_ests.append(x_ests)
        all_us.append(us)

        if not approach == "nominal":
            if approach == "robust":
                outlier_ratio = compute_outlier_ratio_vp(
                    all_xs, all_zs, mpc_controller.poly_tracks, scale=1, last_dim=3
                )
            else:
                outlier_ratio = compute_outlier_ratio_vp(
                    all_xs, all_zs, mpc_controller.vp_tracks, scale, last_dim=3
                )
        else:
            outlier_ratio = 0
            print("outlier ratio:", outlier_ratio)

        # save data file
        all_xs_array = np.asarray(all_xs)
        all_zs_array = np.asarray(all_zs)
        all_x_ests_array = np.asarray(all_x_ests)
        all_us_array = np.asarray(all_us)
        if approach == "robust":
            vp_ests_array = np.asarray(mpc_controller.poly_ests)
            vp_trackes_array = np.asarray(mpc_controller.poly_tracks)
        else:
            vp_ests_array = np.asarray(mpc_controller.vp_ests)
            vp_trackes_array = np.asarray(mpc_controller.vp_tracks)
        np.save("data/psp/" + approach + "_all_xs.npy", all_xs_array)
        np.save("data/psp/" + approach + "_all_zs.npy", all_zs_array)
        np.save("data/psp/" + approach + "_all_x_ests.npy", all_x_ests_array)
        np.save("data/psp/" + approach + "_all_us.npy", all_us_array)
        np.save("data/psp/" + approach + "_vp_tracks.npy", vp_trackes_array)
        np.save("data/psp/" + approach + "_vp_ests.npy", vp_ests_array)

    # draw constraints
    min_z_0 = np.min(np.array(all_zs)[:, :, 0])
    max_z_0 = np.max(np.array(all_zs)[:, :, 0])
    if env.constraint_cfg["circle"]:
        for i in range(len(env.constraint_cfg["mean"])):
            center = np.array(env.constraint_cfg["mean"][i])
            cov = env.constraint_cfg["cov"][i]
            draw_ellipsoid_2d(ax, center, cov, color=palettes.tue_plot[2], alpha=1.0)
    if env.constraint_cfg["polygon"]:
        # if env.constraint_cfg['a'][0][1]==0:
        draw_polygon_from_lines(
            ax,
            env.constraint_cfg["a"],
            env.constraint_cfg["b"],
            [
                min_z_0 * env.cfg["constraints"]["length_scale"],
                max_z_0 * env.cfg["constraints"]["length_scale"],
            ],
            color=palettes.tue_plot[2],
        )
        # draw_polygon_from_lines(ax, env.constraint_cfg['a'], env.constraint_cfg['b'], [min_z_0-0.2, max_z_0+0.2], color=palettes.tue_plot[2])
    if env.constraint_cfg["exponential"]:
        shift = env.constraint_cfg["shift"]
        radius = env.constraint_cfg["radius"]
        draw_exponential_constraints(
            ax, shift, radius, [min_z_0, max_z_0], color=palettes.tue_plot[2]
        )
    if env.constraint_cfg["funnel"]:
        draw_funnel_constraints_2d(
            ax,
            [min_z_0 + 0.1, max_z_0],
            100,
            env.constraint_cfg["funnel"],
            color=palettes.tue_plot[2],
        )

    # visualize samples
    if not approach == "nominal":
        if approach == "sub-Gaussian" or approach == "Gaussian" or approach == "DR":
            draw_trajectories_vp(
                ax,
                all_xs,
                all_zs,
                all_x_ests,
                mpc_controller.vp_ests,
                mpc_controller.vp_tracks,
                scale=scale,
                alpha=0.2,
            )
        elif mpc_controller.mpc_params["tube"]["robust_approach"] == "ellipsoid":
            draw_trajectories_vp(
                ax,
                all_xs,
                all_zs,
                all_x_ests,
                mpc_controller.poly_ests,
                mpc_controller.poly_tracks,
                scale=scale,
                alpha=0.2,
            )
        elif mpc_controller.mpc_params["tube"]["robust_approach"] == "polygon":
            draw_trajectories_robust(
                ax,
                all_xs,
                all_zs,
                all_x_ests,
                mpc_controller.poly_ests,
                mpc_controller.poly_track_ests,
                mpc_controller.poly_track_trues,
                alpha=0.2,
            )
    else:
        all_xs = np.asarray(all_xs)
        all_zs = np.asarray(all_zs)
        all_x_ests = np.asarray(all_x_ests)
        draw_nominal_traj_and_samples(ax, all_xs)

    unsafe_ratio = compute_unsafe_ratio(all_xs, env)

    ave_unsafe_steps = compute_average_fail_number(all_xs, env)

    average_cost = compute_average_cost(
        all_xs,
        all_us,
        mpc_controller.Q,
        mpc_controller.R,
        np.array(mpc_controller.mpc_params["env"]["goal_loc"]),
    )

    return outlier_ratio, unsafe_ratio, ave_unsafe_steps, average_cost


if __name__ == "__main__":
    # set the random seed

    num_env = 1
    env_id = "PedicleScrewPlacement:simple-v0"
    cfg_file = "cfgs/envs/psp.yaml"
    cfg = YAML().load(open(cfg_file, "r"))
    env = gym.make(env_id, cfg_file=cfg_file)

    # test trajectory propagation
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(figsizes.neurips2022(nrows=1, ncols=3))
    plt.rcParams["font.size"] = 15
    plt.rcParams["figure.figsize"] = [17, 17]
    fig, axes = plt.subplots(2, 3)
    fig.tight_layout()

    unsafe_ratio = 0.05

    mpc_cfg = YAML().load(open("cfgs/controller/tube_ellipsoid_params.yaml"))

    # set the initial state
    mean_x_0 = np.array(cfg["start_loc"])
    goal = np.array(cfg["goal_loc"])

    print("mean_x_0:", mean_x_0)

    out_robust = 0
    cost_robust = 0
    num_robust = 0
    safe_robust = 0
    np.random.seed(3)
    # out_robust, safe_robust, num_robust, cost_robust = test_trajectory_mpc('robust', mean_x_0, goal, env, mpc_cfg, unsafe_ratio, 200, 1000, axes[0, 1])

    out_DR = 0
    cost_DR = 0
    num_DR = 0
    safe_DR = 0
    np.random.seed(3)
    out_DR, safe_DR, num_DR, cost_DR = test_trajectory_mpc(
        "DR", mean_x_0, goal, env, mpc_cfg, unsafe_ratio, 200, 1000, axes[0, 2]
    )

    np.random.seed(3)
    out_subGau, safe_subGau, num_subGau, cost_subGau = test_trajectory_mpc(
        "sub-Gaussian",
        mean_x_0,
        goal,
        env,
        mpc_cfg,
        unsafe_ratio,
        200,
        1000,
        axes[0, 0],
    )

    out_nominal = 0
    cost_nominal = 0
    num_nominal = 0
    np.random.seed(3)

    print("outlier ratio robust:", out_robust)
    print("outlier ratio sub-Gaussian:", out_subGau)
    print("outlier ratio nominal:", out_nominal)
    print("outlier ratio DR:", out_DR)
    print("")
    print("unsafe ratio robust:", safe_robust)
    print("unsafe ratio sub-Gaussian:", safe_subGau)
    print("unsafe ratio nominal:", 0)
    print("unsafe ratio DR:", safe_DR)
    print("average cost robust:", cost_robust)
    print("average cost sub-Gaussian:", cost_subGau)
    print("average cost nominal:", cost_nominal)
    print("average cost DR:", cost_DR)
    print("")
    print("average unsafe steps robust:", num_robust)
    print("average unsafe steps sub-Gaussian:", num_subGau)
    print("average unsafe steps nominal:", num_nominal)
    print("average unsafe steps DR:", num_DR)

    plt.show()

    fig.savefig("images/mpc_2d_" + str(env.cfg["name"]) + ".pdf", dpi=250)
