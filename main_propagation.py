from matplotlib import patches
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
from tueplots import figsizes, cycler
from main_compare_cp import sample_trajectory
import time


def test_trajectory_vp_propagation(
    flag: int,
    mean_x_0: np.ndarray,
    env: LinearEnv,
    controller: DummyController,
    prob: float,
    num_steps: int,
    num_samples: int,
    ax: matplotlib.axes.Axes,
):
    """
    flag: 0 for variance proxy, 1 for covariance matrix, 2 for conservative covariance matrix bound
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
    for _ in range(num_samples):
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
        for i in range(2 * num_samples):
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

    vp_w_gau = dists.T @ dists / dists.shape[0]
    vp_e_gau = meas.T @ meas / dists.shape[0]
    vp_0_gau = init_states.T @ init_states / dists.shape[0]
    # estimate variance proxy
    if flag == 0:
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
        vp_0 = vp_0_gau
        vp_w = vp_w_gau
        vp_e = vp_e_gau

    # compute confidence interval
    if flag == 0:
        m = optimize_confidence_interval(prob, env.A.shape[0])[0]
        print("m:", m)  # from varaice proxy to bound
        # get bound scale
        scale = get_bound_scale_given_probability_from_m(prob, m, env.A.shape[0])
    elif flag == 1:
        scale = gaussian_scale_from_prob(prob, env.A.shape[0], [1e-7, 10], 10000)
        print("m_std:", scale)
    else:
        if "state_variant" in env.noise_cfg:
            vp_w_gau = dists_2.T @ dists_2 / dists_2.shape[0]
            vp_e_gau = meas_2.T @ meas_2 / dists_2.shape[0]
            vp_0_gau = init_states_2.T @ init_states_2 / dists_2.shape[0]
            vp_0 = vp_0_gau
            vp_w = vp_w_gau
            vp_e = vp_e_gau
        scale = np.sqrt(env.A.shape[0] / prob)

    time_start = time.time()
    # rollout kalman gain
    Ls = kalman_filter_gain_rollout(
        vp_0_gau, env.A, env.C, vp_w_gau, vp_e_gau, num_steps
    )
    # uncertainty quantification
    vp_ests, vp_tracks = vp_propagation_est_track(
        vp_0, vp_w, vp_e, env.A, env.B, env.C, controller.K, Ls, num_steps
    )
    time_end = time.time()
    print("Time for propagation: ", time_end - time_start)
    # apply control to sample trajectories
    all_zs = []
    all_xs = []
    all_x_ests = []
    for _ in range(num_samples):
        # reset the env
        x, y = env.reset(mean_x_0)
        z = mean_x_0
        x_est = z + Ls[0] @ (y - env.C @ z)
        s_x = vp_0
        # apply control
        zs = [z]
        xs = [x]
        x_ests = [x_est]
        for t in range(num_steps):
            u, v = controller.policy(x_est, z)
            x, y = env.step(x, u)
            z = env.A @ z + env.B @ v
            x_est, s_x = kalman_filter_fix_gain(
                x_est, u, s_x, y, env.A, env.B, env.C, Ls[t + 1], vp_w, vp_e
            )
            zs.append(z)
            xs.append(x)
            x_ests.append(x_est)
        all_zs.append(zs)
        all_xs.append(xs)
        all_x_ests.append(x_ests)

    outlier_ratio = compute_max_outlier_ratio_vp(all_xs, all_zs, vp_tracks, scale)
    # compute ious
    direction = np.zeros((env.A.shape[0],))
    direction[0] = 1
    err_sizes = compute_error_size_vp(vp_tracks, scale, direction)
    # visualize samples
    if flag == 0:
        ax.plot(err_sizes, label="Sub-Gaussian", linewidth=3)
    elif flag == 1:
        if "state_variant" in env.noise_cfg:
            ax.plot(err_sizes, label="Gaussian", linestyle="--")
        else:
            ax.plot(err_sizes, label="Gaussian")

    else:
        ax.plot(err_sizes, label="DR")
    return outlier_ratio, err_sizes


def test_trajectory_robust(
    approach: str,
    mean_x_0: np.ndarray,
    env: LinearEnv,
    controller: DummyController,
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
        dists = np.asarray(dists_2)
        meas = np.asarray(meas_2)
        init_states = np.asarray(init_states_2)

    if approach == "polygon":
        poly_w = compute_bound_points(dists)
        poly_e = compute_bound_points(meas)
        poly_0 = compute_bound_points(init_states)
    else:
        poly_w = np.max(np.linalg.norm(dists, axis=1))
        poly_e = np.max(np.linalg.norm(meas, axis=1))
        poly_0 = np.max(np.linalg.norm(init_states, axis=1))
        # get squared norm for propagation
        poly_w = poly_w**2 * np.eye(env.A.shape[0])
        poly_e = poly_e**2 * np.eye(env.C.shape[0])
        poly_0 = poly_0**2 * np.eye(env.A.shape[0])
    # estimate variance proxy
    vp_w = dists.T @ dists / dists.shape[0]
    vp_e = meas.T @ meas / dists.shape[0]
    vp_0 = init_states.T @ init_states / dists.shape[0]
    print("vp of dist: ", vp_w)
    print("vp of meas:", vp_e)
    print("vp of init:", vp_0)

    time_start = time.time()
    # rollout kalman gain
    Ls = kalman_filter_gain_rollout(vp_0, env.A, env.C, vp_w, vp_e, num_steps)
    # propagate robust sets
    if approach == "polygon":
        poly_ests, poly_track_ests, poly_track_trues = robust_propagation(
            poly_0, poly_w, poly_e, env.A, env.B, env.C, controller.K, Ls, num_steps
        )
    else:
        poly_ests, poly_tracks = robust_ellipsoid_propagation(
            poly_0, poly_w, poly_e, env.A, env.B, env.C, controller.K, Ls, num_steps
        )
    time_end = time.time()
    print("Time for propagation: ", time_end - time_start)

    time_start = time.time()
    # apply control to sample trajectories
    all_zs, all_xs, all_x_ests = sample_trajectory(
        env, controller, num_steps, num_samples, mean_x_0, vp_0, Ls, vp_w, vp_e
    )
    # conformal prediction
    cp_bounds = conformal_predicton_trajectory(all_xs, all_zs, num_steps, prob)
    # visualize samples
    time_end = time.time()
    print("Time for cp: ", time_end - time_start)

    # compute outliers
    direction = np.zeros((env.A.shape[0],))
    direction[0] = 1
    if approach == "polygon":
        outlier_ratio = compute_max_outlier_ratio_robust(
            all_xs, all_zs, poly_track_trues
        )
        err_sizes = compute_error_size_robust(poly_track_trues, direction)
    else:  # use elliposoid to compute outliers
        outlier_ratio = compute_max_outlier_ratio_vp(all_xs, all_zs, poly_tracks, 1.0)
        err_sizes = compute_error_size_vp(poly_tracks, 1.0, direction)

    ax.plot(err_sizes, label="Robust")
    sample_err_sizes = compute_error_size_vp(cp_bounds, 1.0, direction)
    ax.plot(sample_err_sizes, label="Samples")

    # cps:
    v_fake = -(np.random.rand(T, K.shape[0]) + 0.4) * np.array(env.u_max) * 1.5
    controller_fake = DummyController(K, L, v_fake)
    all_zs_fake, all_xs_fake, all_x_ests_fake = sample_trajectory(
        env, controller_fake, num_steps, num_samples, mean_x_0, vp_0, Ls, vp_w, vp_e
    )
    cp_bounds_fake = conformal_predicton_trajectory(
        all_xs_fake, all_zs_fake, num_steps, prob
    )
    sample_err_sizes_fake = compute_error_size_vp(cp_bounds_fake, 1.0, direction)
    cp_outlier_ratio = compute_max_outlier_ratio_vp(all_xs, all_zs, cp_bounds_fake, 1.0)
    print("cp outlier ratio: ", cp_outlier_ratio)
    return cp_outlier_ratio, outlier_ratio, err_sizes


def make_ellipse(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    size = min(width, height)
    center = 1.0 * size, 0.3 * size
    p = patches.Ellipse(center, size * 1.0, size * 1.0)
    return p


if __name__ == "__main__":
    bundle = bundles.icml2024()
    bundle["legend.fontsize"] = 8
    bundle["font.size"] = 11
    plt.rcParams.update(bundle)
    plt.rcParams.update(figsizes.icml2024_half())
    plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
    plt.rcParams["figure.figsize"] = [3.5, 2.5]
    outside_ratio = 0.05
    # set the env
    cfg = YAML().load(open("cfgs/envs/mass_spring_damper.yaml", "r"))
    env = LinearEnv(cfg)
    if env.A.shape[0] == 2:
        approach = "polygon"
    else:
        approach = "ellipsoid"
    # set the controller
    num_samples = 2000  # 100000
    P, K = solve_LQR(env.A, env.B, env.Q, env.R)
    L = np.eye(2)
    T = 60
    v_seq = (np.random.rand(T, K.shape[0]) + 0.1) * np.array(env.u_max) * 0.1
    controller = DummyController(K, L, v_seq)
    mean_x_0 = np.array(env.cfg["start_loc"])
    noise_list = ["bounded_laplace", "skew_normal", "gaussian", "uniform", "standard_t"]
    species = noise_list
    outlier_results = {
        "sub-Gaussian": [],
        "Robust": [],
        "Distributional Robust": [],
        "cp": [],
    }
    err_size_results = {
        "sub-Gaussian": [],
        "Robust": [],
        "Distributional Robust": [],
        "cp": [],
    }
    for noise in noise_list:
        env.noise_cfg["disturbance"]["type"] = noise
        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()

        # sub gaussian
        np.random.seed(3)
        subgau_outlier, subgau_err_sizes = test_trajectory_vp_propagation(
            0, mean_x_0, env, controller, outside_ratio, T, num_samples, ax
        )
        # # gaussian
        # dist robust
        np.random.seed(3)
        dist_outlier, dist_err_sizes = test_trajectory_vp_propagation(
            2, mean_x_0, env, controller, outside_ratio, T, num_samples, ax
        )

        np.random.seed(3)
        cp_outlier, robust_outlier, robust_err_sizes = test_trajectory_robust(
            approach, mean_x_0, env, controller, outside_ratio, T, num_samples, ax
        )

        ax.legend(loc="best", ncol=2)
        ax.set_yscale("log")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Bound size")
        ax.set_ylim([0, 100])  # 5 100

        fig.savefig(
            "images/prop_size" + cfg["noise"]["disturbance"]["type"] + ".pdf",
            dpi=300,
        )
        outlier_results["sub-Gaussian"].append(1 - subgau_outlier)
        outlier_results["Robust"].append(1 - robust_outlier)
        outlier_results["Distributional Robust"].append(1 - dist_outlier)
        outlier_results["cp"].append(1 - cp_outlier)
        err_size_results["sub-Gaussian"].append(np.mean(np.asarray(subgau_err_sizes)))
        err_size_results["Robust"].append(np.mean(np.asarray(robust_err_sizes)))
        err_size_results["Distributional Robust"].append(
            np.mean(np.asarray(dist_err_sizes))
        )
        err_size_results["cp"].append(np.mean(np.asarray(robust_err_sizes)))
        print("Sub Gaussian Outliers:", subgau_outlier)
        print("Robust Outliers:", robust_outlier)
        print("Distributional Robust Outliers:", dist_outlier)
        print("CP Outliers:", cp_outlier)
        print("Sub Gaussian err_sizes:", np.mean(np.asarray(subgau_err_sizes)))
        print("Robust err_sizes:", np.mean(np.asarray(robust_err_sizes)))
        print("Distributional Robust err_sizes:", np.mean(np.asarray(dist_err_sizes)))
    fig, axes = plt.subplots(2, 1)
    draw_bars_from_dict(axes[0], outlier_results, species, "Inlier ratios")
    axes[0].axhline(y=0.99, color="blue", linestyle=":", linewidth=2)
    axes[0].set_ylim([0.9, 1.03])
    draw_bars_from_dict(axes[1], err_size_results, species, "err_size ratios")
    fig.savefig("images/stable_numerical.pdf", dpi=300)
    for key in outlier_results.keys():
        print(key, "ave outliear ratios", np.mean(np.array(outlier_results[key])))
        print(key, "ave err_size with cp", np.mean(np.array(err_size_results[key])))
