from matplotlib import patches
from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt
import numpy as np
from uncertainty.estimate_variance_proxy import *
from uncertainty.optimize_confidence_interval import *
from uncertainty.compute_confidence_bound import *
from visualize import *
from scipy.stats import skewnorm
from tueplots import bundles
from tueplots.constants.color import palettes
from tueplots import figsizes


def calibrate_samples(samples, prob, ax):
    """
    sample_list: (N, n)
    """
    n = samples.shape[-1]
    print("n:", n)
    # first pca
    mean = np.mean(samples, axis=0)
    centered_samples = samples - mean
    cov = centered_samples.T @ centered_samples  # (d, d)
    L = np.linalg.cholesky(cov)
    n_samples = np.dot(centered_samples, np.linalg.inv(L).T)
    print("mean:", mean)
    # estimate c4 from samples
    c4 = find_C4(n_samples[:, :], [0.00, 1], 10000)
    print("c4:", c4**2 * L @ np.eye(n) @ L.T)
    # estimate vp from samples
    vp, params = estimate_variance_proxy(n_samples[:, :])
    print("vp:", vp**2 * L @ np.eye(n) @ L.T)
    print("lambda:", params)
    # get variance
    std = np.std(n_samples[:, :], axis=0)[0]
    print("std:", std**2 * L @ np.eye(n) @ L.T)
    # solve the normalized confidence interval
    m = optimize_confidence_interval(prob, n)[0]
    m_10 = optimize_confidence_interval(prob * 10, n)[0]
    m_std = gaussian_scale_from_prob(prob, n, [0.0, 10.0], 10000)
    m_std_10 = gaussian_scale_from_prob(prob * 10, n, [0.0, 10.0], 10000)
    print("m:", m)
    print("m_std", m_std)
    # get bounds with high probability
    bound_c4 = get_bound_given_probability_from_c4(prob, c4)
    bound_vp = get_bound_given_probability_from_vp_and_m(prob, vp, m, n)
    bound_vp_10 = get_bound_given_probability_from_vp_and_m(prob * 10, vp, m_10, n)
    bound_std = m_std * std
    bound_std_10 = m_std_10 * std
    # get matrices
    vp_ellipsoid = bound_vp**2 * L @ np.eye(n) @ L.T
    c4_ellipsoid = bound_c4**2 * L @ np.eye(n) @ L.T
    std_ellipsoid = bound_std**2 * L @ np.eye(n) @ L.T
    vp_ellipsoid_10 = bound_vp_10**2 * L @ np.eye(n) @ L.T
    std_ellipsoid_10 = bound_std_10**2 * L @ np.eye(n) @ L.T
    print("vp_ellip:", vp_ellipsoid)
    print("c4_ellip:", c4_ellipsoid)
    print("std_ellip:", std_ellipsoid)
    print(samples.shape)
    palette = plt.cm.tab10.colors
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        label="samples",
        s=0.5,
        color=palette[0],
        rasterized=True,
    )
    draw_ellipsoid_2d(
        ax,
        mean,
        vp_ellipsoid[0:2, 0:2],
        palettes.tue_plot[0],
        1,
        "sub-Gaussian bound 0.01",
    )
    draw_ellipsoid_2d(
        ax,
        mean,
        vp_ellipsoid_10[0:2, 0:2],
        palettes.tue_plot[6],
        1,
        "sub-Gaussian bound 0.1",
    )
    draw_ellipsoid_2d(
        ax,
        mean,
        std_ellipsoid[0:2, 0:2],
        palettes.tue_plot[2],
        1,
        "Gaussian bound 0.01",
    )
    draw_ellipsoid_2d(
        ax,
        mean,
        std_ellipsoid_10[0:2, 0:2],
        palettes.tue_plot[5],
        1,
        "Gaussian bound 0.1",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # second histogram
    # ax2: density of sub gaussian, and histogram of samples
    norms = np.linalg.norm(n_samples, axis=-1)
    max_norm = np.max(norms)
    norm_coords = np.linspace(0, max_norm * 1.2, 100)
    n_norm_coords_vp = norm_coords / vp
    # compute outlier of covariance
    all_cov = samples @ np.linalg.inv(std_ellipsoid) @ samples.T  # (N, N)
    diag = np.diagonal(all_cov)
    print("std_out 0.01:", np.sum(diag > 1) / samples.shape[0])
    all_cov = samples @ np.linalg.inv(std_ellipsoid_10) @ samples.T  # (N, N)
    diag = np.diagonal(all_cov)
    print("std_out 0.1:", np.sum(diag > 1) / samples.shape[0])
    all_vp = samples @ np.linalg.inv(vp_ellipsoid) @ samples.T  # (N, N)
    diag = np.diagonal(all_vp)
    print("vp_out 0.01:", np.sum(diag > 1) / samples.shape[0])
    all_vp = samples @ np.linalg.inv(vp_ellipsoid_10) @ samples.T  # (N, N)
    diag = np.diagonal(all_vp)
    print("vp_out 0.1:", np.sum(diag > 1) / samples.shape[0])
    # tail bounds
    vp_cdf = tail_bound_from_vp_and_m(n_norm_coords_vp, m, n)
    fig1, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    # invert cdf
    ax1.hist(
        norms,
        30,
        density=True,
        cumulative=-1,
        label="Reversed cdf",
        color=palettes.tue_plot[3],
    )
    ax1.plot(
        norm_coords,
        vp_cdf,
        color=palettes.tue_plot[5],
        label="Sub-Gaussian tail bound",
        linewidth=2,
    )
    ax1.legend()
    fig1.savefig("images/dists_tail" + ".pdf")
    ax.set_axis_off()
    return vp_ellipsoid


def make_ellipse(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    size = min(width, height)
    center = 1. * size, 0.3 * size
    p = patches.Ellipse(center, size * 1., size * 1.)
    return p


if __name__ == "__main__":
    bundle = bundles.icml2024()
    bundle["legend.fontsize"] = 8
    bundle["font.size"] = 11
    plt.rcParams.update(bundle)
    plt.rcParams.update(
        figsizes.icml2024_full(rel_width=1.0, height_to_width_ratio=0.43)
    )
    fig, axes = plt.subplots(1, 6)
    np.random.seed(0)
    # test student_t
    samples = []
    for i in range(5000):
        n = np.random.standard_t(df=5, size=(1, 2))
        while np.linalg.norm(n) > 5.0:
            n = np.random.standard_t(df=5, size=(1, 2))
        samples.append(n)
    samples = np.concatenate(samples, axis=0)
    vp = calibrate_samples(samples, 0.01, axes[0])
    axes[0].set_title("Student-T", pad=-10)
    # # test normal
    samples = np.random.normal(0.0, 1.0, (5000, 2))
    vp = calibrate_samples(samples, 0.01, axes[1])
    axes[1].set_title("Normal", pad=-10)
    # test laplace
    # # test star
    samples = []
    for i in range(5000):
        sample = np.random.laplace(0.0, 1.0, (1, 2))
        while np.linalg.norm(sample) > 5.0:
            sample = np.random.laplace(0.0, 1.0, (1, 2))
        samples.append(sample)
    samples = np.concatenate(samples, axis=0)
    vp = calibrate_samples(samples, 0.01, axes[2])
    axes[2].set_title("Laplace", pad=-10)
    # test uniform
    samples = np.random.uniform(-1.0, 1.0, (5000, 2))
    vp = calibrate_samples(samples, 0.01, axes[3])
    axes[3].set_title("Uniform", pad=-10)
    # # test star
    samples = []
    for i in range(5000):
        sample = np.random.laplace(0.0, 1.0, (1, 2))
        while (np.abs(sample) > 0.5).all(axis=-1) or np.linalg.norm(sample) > 5.0:
            sample = np.random.laplace(0.0, 1.0, (1, 2))
        samples.append(sample)
    samples = np.concatenate(samples, axis=0)

    # test skew normal
    a = 10.0
    n = skewnorm.rvs(a, scale=0.01, size=(5000, 2))
    delta = a / np.sqrt(1 + a**2)
    mean = 0.01 * delta * np.sqrt(2 / np.pi)
    n -= mean
    vp = calibrate_samples(n, 0.01, axes[4])
    axes[4].set_title("Skew Normal", pad=-10)
    lines, labels = axes[0].get_legend_handles_labels()

    # registration errors
    samples = np.load("data/psp/0928/est_errs.npy")[:, [0, 1]]/1000
    vp = calibrate_samples(samples, 0.01, axes[5])
    axes[5].set_title("Registration Error", pad=-10)
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        frameon=False,
        handletextpad=-0.4,
        handler_map={patches.Ellipse: HandlerPatch(patch_func=make_ellipse)},
    )
    fig.savefig("images/dists" + ".pdf", dpi=300)
