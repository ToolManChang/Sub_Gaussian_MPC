from matplotlib import patches
from matplotlib.legend_handler import HandlerPatch
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
import gym
import matplotlib.patches as patches
from tueplots import bundles, figsizes


def make_ellipse(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    size = min(width, height)
    center = 1.0 * size, 0.3 * size
    p = patches.Ellipse(center, size * 1.0, size * 1.0)
    return p


def test_trajectory_mpc(
    approach: str,
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
    for i in range(num_samples):
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
    if approach == "sub-Gaussian":
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
        vp_w = dists.T @ dists / dists.shape[0]
        vp_e = meas.T @ meas / meas.shape[0]
        vp_0 = init_states.T @ init_states / init_states.shape[0]

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
            prob, env.cfg["constraints"]["num_dim"], [1e-8, 10], 100000
        )
        cfg["approach"] = "Gaussian"
        print("scale", scale)
    elif approach == "DR":
        scale = np.sqrt(env.cfg["constraints"]["num_dim"] / prob)
        cfg["approach"] = "DR"
    else:
        scale = 1
        cfg["approach"] = "nominal"
        print("scale", scale)

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
        # estimate variance proxy
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
    num_y_below = 0
    for sample in range(int(num_samples / 10)):
        # reset the env
        x, y = env.reset(mean_x_0)
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
            print("y below", num_y_below)
        for t in range(num_steps):
            x, y = env.step(x, u)
            if (
                "state_variant" in env.noise_cfg
                and x[1] < env.noise_cfg["state_variant"]["y_threshold"]
            ):
                num_y_below += 1

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

        all_zs.append(zs)
        all_xs.append(xs)
        all_x_ests.append(x_ests)
        all_us.append(us)

    all_zs = np.asarray(all_zs)
    all_xs = np.asarray(all_xs)
    all_x_ests = np.asarray(all_x_ests)
    all_us = np.asarray(all_us)

    if (
        approach == "sub-Gaussian"
        or approach == "Gaussian"
        or approach == "nominal"
        or approach == "DR"
    ):
        (
            mean_ious,
            outlier,
            max_unsafe_steps,
            ave_unsafe_steps,
            average_cost,
            cp_outlier_ratio,
        ) = analyze_data(
            env,
            env.cfg,
            all_xs,
            all_zs,
            all_x_ests,
            all_us,
            mpc_controller.vp_ests,
            mpc_controller.vp_tracks,
            scale,
            prob,
            ax,
            if_plot,
        )

    elif mpc_controller.mpc_params["tube"]["robust_approach"] == "ellipsoid":
        (
            mean_ious,
            outlier,
            max_unsafe_steps,
            ave_unsafe_steps,
            average_cost,
            cp_outlier_ratio,
        ) = analyze_data(
            env,
            env.cfg,
            all_xs,
            all_zs,
            all_x_ests,
            all_us,
            mpc_controller.poly_ests,
            mpc_controller.poly_tracks,
            scale,
            prob,
            ax,
            if_plot,
        )

    return (
        mean_ious,
        outlier,
        max_unsafe_steps,
        ave_unsafe_steps,
        average_cost,
        cp_outlier_ratio,
    )


def analyze_data(
    env,
    env_cfg,
    all_xs,
    all_zs,
    all_x_ests,
    all_us,
    vp_ests,
    vp_tracks,
    scale,
    prob,
    ax,
    if_plot=True,
):

    if if_plot:
        # visualize samples
        draw_trajectories_vp(
            ax, all_xs, all_zs, all_x_ests, vp_ests, vp_tracks, scale=scale, alpha=0.2
        )

        # draw constraints
        min_z_0 = np.min(np.array(all_zs)[:, :, 0])
        max_z_0 = np.max(np.array(all_zs)[:, :, 0])
        if env.constraint_cfg["circle"]:
            for i in range(len(env.constraint_cfg["mean"])):
                center = np.array(env.constraint_cfg["mean"][i])
                cov = env.constraint_cfg["cov"][i]
                draw_ellipsoid_2d(
                    ax, center, cov, color=palettes.tue_plot[2], alpha=1.0
                )
        if env.constraint_cfg["polygon"]:
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
        if env.constraint_cfg["exponential"]:
            shift = env.constraint_cfg["shift"]
            radius = env.constraint_cfg["radius"]
            draw_exponential_constraints(
                ax, shift, radius, [min_z_0, max_z_0], color=palettes.tue_plot[2]
            )
        if env.constraint_cfg["funnel"]["if_funnel"]:
            draw_funnel_constraints_2d(
                ax,
                [min_z_0 + 0.1, max_z_0],
                100,
                env.constraint_cfg["funnel"],
                color=palettes.tue_plot[2],
            )

        # draw goal
        ax.scatter(
            env_cfg["goal_loc"][0],
            env_cfg["goal_loc"][1],
            color="gray",
            s=60,
            marker="*",
            label="Goal",
            zorder=200,
        )

    # numerical results
    # compute outliers
    unsafe_ratio = compute_unsafe_ratio(all_xs, env)

    ave_unsafe_steps = compute_average_fail_number(all_xs, env)
    max_unsafe_steps = compute_max_fail_number(all_xs, env)

    Q = np.diag(np.array(env_cfg["Q"]))
    R = np.diag(np.array(env_cfg["R"]))
    average_cost = compute_average_cost(
        all_xs, all_us, Q, R, np.array(env_cfg["goal_loc"])
    )
    outlier_ratio = compute_max_outlier_ratio_vp(
        all_xs, all_zs, vp_tracks, scale, last_dim=3
    )

    # conformal prediction
    cp_bounds = conformal_predicton_trajectory(
        all_xs, all_zs, np.array(all_zs).shape[1] - 1, prob, last_dim=3
    )
    cp_outlier_ratio = compute_max_outlier_ratio_vp(
        all_xs, all_zs, cp_bounds, 1.0, last_dim=3
    )
    ious = compute_iou_vp(vp_tracks, scale, cp_bounds, last_dim=3)
    mean_ious = np.mean(ious)

    return (
        mean_ious,
        outlier_ratio,
        max_unsafe_steps,
        ave_unsafe_steps,
        average_cost,
        cp_outlier_ratio,
    )


if __name__ == "__main__":
    bundle = bundles.icml2024()
    bundle["legend.fontsize"] = 11
    bundle["font.size"] = 50
    plt.rcParams.update(bundle)
    plt.rcParams.update(figsizes.icml2024_full(nrows=1))
    # plt.rcParams['figure.figsize'] = [10.5, 7.0]
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
    env_list = ["rocket.yaml"]

    # test trajectory propagation
    fig, axes = plt.subplots(1, 2)
    unsafe_ratio = 0.05

    # read data for psp experiment
    env_id = "PedicleScrewPlacement:simple-v0"
    cfg_file = "cfgs/envs/psp.yaml"
    cfg = YAML().load(open(cfg_file, "r"))
    env = gym.make(env_id, cfg_file=cfg_file)

    i = 0
    if_plot = True
    for approach in ["sub-Gaussian", "DR"]:  # ['Gaussian', 'sub-Gaussian', 'DR']:
        all_xs = np.load("data/psp/1113/" + approach + "_all_xs.npy")
        all_zs = np.load("data/psp/1113/" + approach + "_all_zs.npy")
        all_x_ests = np.load("data/psp/1113/" + approach + "_all_x_ests.npy")
        all_us = np.load("data/psp/1113/" + approach + "_all_us.npy")
        vp_tracks = np.load("data/psp/1113/" + approach + "_vp_tracks.npy")
        vp_ests = np.load("data/psp/1113/" + approach + "_vp_ests.npy")

        if approach == "sub-Gaussian":
            m = optimize_confidence_interval(unsafe_ratio, env.A.shape[0])[0]
            print("m:", m)  # from varaice proxy to bound

            # get bound scale
            # gaussian and sub ga
            s = get_bound_scale_given_probability_from_m(
                unsafe_ratio, m, env.A.shape[0]
            )
            cfg["approach"] = "sub-Gaussian"
        elif approach == "Gaussian":
            s = gaussian_scale_from_prob(
                unsafe_ratio, env.A.shape[0], [1e-7, 10], 10000
            )
            cfg["approach"] = "Gaussian"
            print("scale", s)
        elif approach == "robust":
            s = 1
            cfg["approach"] = "robust"
            print("scale", s)
        elif approach == "DR":
            s = np.sqrt(env.A.shape[0] / unsafe_ratio)
        else:
            s = 1
            cfg["approach"] = "nominal"
            print("scale", s)

        (
            mean_iou_subGau,
            out_subGau,
            safe_subGau,
            num_subGau,
            cost_subGau,
            cp_out_subGau,
        ) = analyze_data(
            env,
            cfg,
            all_xs,
            all_zs,
            all_x_ests,
            all_us,
            vp_ests,
            vp_tracks,
            s,
            unsafe_ratio,
            axes[i],  # axes[0, i],
            if_plot,
        )

        print("mean iou " + approach, mean_iou_subGau)
        print("outlier ratio " + approach, out_subGau)
        print("max unsafe steps " + approach, safe_subGau)
        print("average cost " + approach, cost_subGau)
        print("average unsafe steps " + approach, num_subGau)
        print("cp outlier ratio " + approach, cp_out_subGau)

        # axes[1].set_title('Surgical Planning',fontsize = 10)
        axes[i].set_title(approach, fontsize=12)
        if approach == "sub-Gaussian":
            axes[i].set_title("Sub-Gaussian SMPC (Ellipsoid)", fontsize=12)
        if approach == "DR":
            axes[i].set_title("DR SMPC", fontsize=12)
        axes[i].set_xlabel("x[0]", fontsize=12)
        axes[i].set_ylabel("x[1]", fontsize=12)
        axes[i].set_ylim(-0.02, 0.04)
        i += 1

    # lines, labels = axes[0, 0].get_legend_handles_labels()
    lines, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    plt.show()
    fig.savefig("images/mpc_2d_" + str(env.cfg["name"]) + ".pdf", dpi=250)
