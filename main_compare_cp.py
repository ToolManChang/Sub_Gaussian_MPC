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
from tueplots import bundles
from tueplots import figsizes


def sample_trajectory(
    env, controller, num_steps, num_samples, mean_x_0, vp_0, Ls, vp_w, vp_e
):
    # apply control to sample trajectories
    all_zs = []
    all_xs = []
    all_x_ests = []
    for i in range(num_samples):
        if i % 1000 == 0:
            print(i)
        # reset the env
        x, y = env.reset(mean_x_0)
        z = mean_x_0
        x_est = z + Ls[0] @ (y - env.C @ z)
        s_x = vp_0
        # apply control
        # swap dims
        inds = np.array([1, 0])
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
    return all_zs, all_xs, all_x_ests


def test_trajectory_vp_propagation(
    flag: int,
    mean_x_0: np.ndarray,
    env: LinearEnv,
    controller: DummyController,
    prob: float,
    num_steps: int,
    num_samples: int,
    axes: matplotlib.axes.Axes,
):
    """
    flag: 0 for variance proxy, 1 for covariance matrix, 2 for sub gaussian norm
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
    if env.noise_cfg["time_variant"]["activate"]:
        dists_2 = []
        meas_2 = []
        init_states_2 = []
        interval = env.noise_cfg["time_variant"]["interval"]
        for i in range(44 * num_samples):
            dist_n = env.sample_noise(dist_cfg, env.A.shape[0])
            if (env.sample_count - 1) % (2 * interval) >= interval:
                dists_2.append(dist_n)
            else:
                dists.append(dist_n)
            meas_n = env.sample_noise(meas_cfg, env.C.shape[0])
            if (env.sample_count - 1) % (2 * interval) >= interval:
                meas_2.append(meas_n)
            else:
                meas.append(meas_n)
            init_states_n = env.sample_noise(
                env.noise_cfg["init_state"], env.A.shape[0]
            )
            if (env.sample_count - 1) % (2 * interval) >= interval:
                init_states_2.append(init_states_n)
            else:
                init_states.append(
                    init_states_n
                )
    else:
        for i in range(10 * num_samples):
            dists.append(env.sample_noise(dist_cfg, env.A.shape[0]))
            meas.append(env.sample_noise(meas_cfg, env.C.shape[0]))
            init_states.append(
                env.sample_noise(env.noise_cfg["init_state"], env.A.shape[0])
            )
    print(env.sample_count)
    # assume scale is smaller than 1, solve this calibration is big enough
    dists = np.asarray(dists)
    meas = np.asarray(meas)
    init_states = np.asarray(init_states)
    # estimate variance proxy
    if flag == 0:
        vp_w = estimate_scaled_covariance_proxy(dists)
        vp_e = estimate_scaled_covariance_proxy(meas)
        vp_0 = estimate_scaled_covariance_proxy(init_states)
        if env.noise_cfg["time_variant"]["activate"]:
            dists_2 = np.asarray(dists_2)
            meas_2 = np.asarray(meas_2)
            init_states_2 = np.asarray(init_states_2)
            vp_w_2 = estimate_scaled_covariance_proxy(dists_2)
            vp_e_2 = estimate_scaled_covariance_proxy(meas_2)
            vp_0_2 = estimate_scaled_covariance_proxy(init_states_2)
            vp_w = (vp_w + vp_w_2)
            vp_e = (vp_e + vp_e_2)
            vp_0 = (vp_0 + vp_0_2)
        print("vp of dist: ", vp_w)
        print("vp of meas:", vp_e)
        print("vp of init:", vp_0)
        # vp_w = vp_w**2 * np.eye(env.A.shape[0])
        # vp_e = vp_e**2 * np.eye(env.C.shape[0])
        # vp_0 = vp_0**2 * np.eye(env.A.shape[0])
    elif flag == 1:
        vp_w = dists.T @ dists / dists.shape[0]
        vp_e = meas.T @ meas / meas.shape[0]
        vp_0 = init_states.T @ init_states / init_states.shape[0]
    else:
        c4_w = find_C4(dists, [1e-7, 10], 10000)
        c4_e = find_C4(meas, [1e-7, 10], 10000)
        c4_0 = find_C4(init_states, [1e-7, 10], 10000)
        vp_w = c4_w**2 * np.eye(env.A.shape[0])
        vp_e = c4_e**2 * np.eye(env.C.shape[0])
        vp_0 = c4_0**2 * np.eye(env.A.shape[0])
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
        scale = np.sqrt(np.log(2 / prob))
    # rollout kalman gain
    Ls = kalman_filter_gain_rollout(vp_0, env.A, env.C, vp_w, vp_e, num_steps)

    # uncertainty quantification
    if flag == 0 or flag == 1:
        vp_ests, vp_tracks = vp_propagation_est_track(
            vp_0, vp_w, vp_e, env.A, env.B, env.C, controller.K, Ls, num_steps # predict one more step
        )
    else:
        vp_ests, vp_tracks = sub_gau_norm_propagation(
            vp_0, vp_w, vp_e, env.A, env.B, env.C, controller.K, Ls, num_steps
        )

    # sample trajectories for calibration
    all_zs, all_xs, all_x_ests = sample_trajectory(
        env, controller, num_steps, num_samples * 2, mean_x_0, vp_0, Ls, vp_w, vp_e
    )
    print(env.sample_count)
    # conformal prediction
    cp_bounds = conformal_predicton_trajectory(all_xs, all_zs, num_steps, prob)

    # sample again for testing
    all_zs, all_xs, all_x_ests = sample_trajectory(
        env, controller, num_steps, num_samples, mean_x_0, vp_0, Ls, vp_w, vp_e
    )
    print(env.sample_count)
    # visualize samples
    draw_trajectories_vp(
        axes[0],
        all_xs,
        all_zs,
        all_x_ests,
        vp_ests,
        vp_tracks,
        cp_bounds[:-1],
        scale=scale,
        alpha=1.0,
    )

    # sample again for testing
    all_zs, all_xs, all_x_ests = sample_trajectory(
        env, controller, num_steps, num_samples, mean_x_0, vp_0, Ls, vp_w, vp_e
    )
    cp_dists = conformal_prediction_state(dists, np.zeros(dists.shape), prob)
    cp_ests = conformal_prediction_state(meas, np.zeros(meas.shape), prob)
    cp_init = conformal_prediction_state(init_states, np.zeros(init_states.shape), prob)
    cp_bounds_ests, cp_bounds_track = sub_gau_norm_propagation(
        cp_init, cp_dists, cp_ests, env.A, env.B, env.C, controller.K, Ls, num_steps
    )
    print(env.sample_count)
    # visualize samples
    draw_trajectories_vp(
        axes[1],
        all_xs,
        all_zs,
        all_x_ests,
        vp_ests,
        vp_tracks,
        cp_bounds_track,
        scale=scale,
        alpha=1.0,
    )
    
    # compute outliers
    outlier_ratio = compute_outlier_ratio_vp(all_xs, all_zs, vp_tracks, scale)
    outlier_cp = compute_outlier_ratio_vp(all_xs, all_zs, cp_bounds, 1.0)
    return outlier_ratio, outlier_cp


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
    plt.rcParams.update(figsizes.icml2024_full(rel_width=1.0))
    outside_ratio = 0.01
    # set the env
    cfg = YAML().load(open("cfgs/envs/mass_spring_damper.yaml", "r"))
    # cfg['noise']['time_variant']['activate'] = True
    env = LinearEnv(cfg)
    if env.A.shape[0] == 2:
        approach = "polygon"
    else:
        approach = "ellipsoid"
    # set the controller
    P, K = solve_LQR(env.A, env.B, env.Q, env.R)
    L = np.eye(2)
    T = 10
    v_seq = (
        (np.random.rand(T, K.shape[0]) + 0.3)
        * np.array(env.u_max)
        * 1.0 #* np.array([8.0, 3.0])
    )
    controller = DummyController(K, L, v_seq)
    # set the initial state
    mean_x_0 = np.zeros((env.A.shape[0],))
    # test trajectory propagation
    fig, axes = plt.subplots(1, 2)
    # sub gaussian
    np.random.seed(1)
    subgau_outlier, subgau_cp = test_trajectory_vp_propagation(
        0, mean_x_0, env, controller, outside_ratio, T, 400, axes
    )
    # gaussian
    print("sub gaussian outlier ratio:", subgau_outlier)
    print("sub gaussian cp outlier ratio:", subgau_cp)
    axes[0].set_title("Conformal prediction with trajectory samples")
    axes[1].set_title("Conformal prediction and propagation with noise samples")
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        handletextpad=-0.4,
        handler_map={patches.Ellipse: HandlerPatch(patch_func=make_ellipse)},
    )
    fig.savefig(
        "images/compare_cp" + cfg["noise"]["disturbance"]["type"] + ".pdf", dpi=300
    )
