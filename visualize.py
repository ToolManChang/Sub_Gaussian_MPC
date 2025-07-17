import numpy as np
import skgeom as sg
import matplotlib.pyplot as plt
from uncertainty.variance_proxy_propagation import *
from tueplots.constants.color import palettes
import pyvista as pv

def draw_ellipsoid_2d(ax, mu, Sigma, color=None, alpha=0.05, label=None):
    """
    draw 2d ellipsoid
    ax: matplotlib axis
    mu: center
    Sigma: covariance matrix
    """
    # get the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(Sigma)
    # get the angle of the major axis
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    # get the width and height of the ellipse
    width, height = 2 * np.sqrt(eigvals)
    # draw the ellipse
    ell = plt.matplotlib.patches.Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=np.degrees(angle),
        edgecolor=color,
        facecolor="none",
        linestyle="-",
        linewidth=1.0,
        alpha=alpha,
        label=label,
        zorder=10,
    )
    ax.add_patch(ell)
    return ell


def draw_polygon_from_lines(ax, a_list, b_list, x0_range, color="b"):
    """
    draw polygon from lines
    ax: matplotlib axis
    a_list: list of a
    b_list: list of b
    x0_range: range of x[0]
    """
    # solve points
    cross_list = []

    n = len(a_list)
    m = 0

    if n == 1:
        a = np.array(a_list[0][0:2])
        b = np.array(b_list[0][0:2])
        if not a[1] == 0:
            cross_list.append(
                np.array([x0_range[0], (b[0] - a[0] * x0_range[0]) / a[1]])
            )
            cross_list.append(
                np.array([x0_range[1], (b[0] - a[0] * x0_range[1]) / a[1]])
            )
        else:
            cross_list.append(
                np.array([(b[0] - a[1] * x0_range[0]) / a[0], x0_range[0]])
            )
            cross_list.append(
                np.array([(b[0] - a[1] * x0_range[1]) / a[0], x0_range[1]])
            )
        m = 2
    else:
        for i in range(n):
            a = np.array(a_list[i][0:2])
            b = np.array(b_list[i][0:2])
            a_next = np.array(a_list[(i + 1) % n])[0:2]
            b_next = np.array(b_list[(i + 1) % n])[0:2]
            if np.linalg.norm(a) == 0 or np.linalg.norm(a_next) == 0:
                continue

            m += 1

            A = np.stack([a, a_next], axis=0)
            B = np.stack([b, b_next], axis=0)

            try:
                cross = np.linalg.solve(A, B)

                cross_list.append(cross)
            except:
                print("non closed polygon")
                cross_list.append(
                    np.array([x0_range[0], (b[0] - a[0] * x0_range[0]) / a[1]])
                )
                cross_list.append(
                    np.array([x0_range[1], (b[0] - a[0] * x0_range[1]) / a[1]])
                )

    for i in range(m):
        ax.plot(
            [cross_list[i][0], cross_list[(i + 1) % m][0]],
            [cross_list[i][1], cross_list[(i + 1) % m][1]],
            color=color,
            linewidth=2,
            zorder=0,
        )


def draw_translated_poly(ax, poly, mu, color="b", alpha=0.05, label=None):
    """
    draw translated polygon
    ax: matplotlib axis
    poly: skgeom.Polygon
    mu: translation vector
    """
    new_poly = sg.Polygon(poly.coords + mu)
    n = poly.coords.shape[0]
    for i in range(n):
        if i == 0:
            ax.plot(
                [new_poly.coords[i, 0], new_poly.coords[(i + 1) % n, 0]],
                [new_poly.coords[i, 1], new_poly.coords[(i + 1) % n, 1]],
                color=color,
                alpha=alpha,
                label=label,
            )
        else:
            ax.plot(
                [new_poly.coords[i, 0], new_poly.coords[(i + 1) % n, 0]],
                [new_poly.coords[i, 1], new_poly.coords[(i + 1) % n, 1]],
                color=color,
                alpha=alpha,
            )


def sort_clockwise(points):
    # Step 1: Calculate the centroid (average of all points)
    centroid = np.mean(points, axis=0)

    # Step 2: Calculate the angle of each point with respect to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Step 3: Sort points by angle in descending order (for clockwise order)
    sorted_points = points[np.argsort(-angles)]

    # Output: sorted points in clockwise order
    return sorted_points


def draw_translated_poly_points(ax, poly_points, mu, color="b", alpha=0.05, label=None):
    """
    draw translated polygon
    ax: matplotlib axis
    poly: skgeom.Polygon
    mu: translation vector
    """
    # first sort clockwise
    points = sort_clockwise(poly_points) + mu
    new_poly = sg.Polygon(points)
    n = poly_points.shape[0]
    for i in range(n):
        if i == 0:
            ax.plot(
                [new_poly.coords[i, 0], new_poly.coords[(i + 1) % n, 0]],
                [new_poly.coords[i, 1], new_poly.coords[(i + 1) % n, 1]],
                color=color,
                alpha=alpha,
                label=label,
            )
        else:
            ax.plot(
                [new_poly.coords[i, 0], new_poly.coords[(i + 1) % n, 0]],
                [new_poly.coords[i, 1], new_poly.coords[(i + 1) % n, 1]],
                color=color,
                alpha=alpha,
            )


def draw_nominal_traj_and_samples(ax, all_xs, all_zs=None, all_x_ests=None):
    flat_all_xs = all_xs.reshape(-1, all_xs.shape[-1])
    # draw the true state
    ax.scatter(
        flat_all_xs[:, 0],
        flat_all_xs[:, 1],
        color=palettes.tue_plot[3],
        s=0.25,
        label="True States",
        rasterized=True,
        zorder=5,
    )
    # draw the nominal state
    if all_zs is not None:
        ax.scatter(
            all_zs[:, :, 0],
            all_zs[:, :, 1],
            c=palettes.tue_plot[1],
            s=0.25,
            label="Nominal States",
            rasterized=True,
            zorder=20,
        )
        for sample in range(all_zs.shape[0]):
            ax.plot(
                all_zs[sample, :, 0],
                all_zs[sample, :, 1],
                color=palettes.tue_plot[1],
                alpha=1.0,
                zorder=20,
            )


def draw_trajectories_vp(
    ax,
    all_xs,
    all_zs,
    all_x_ests,
    vp_est,
    vp_tracks,
    cp_bounds=None,
    scale=1.0,
    alpha=0.05,
):
    """
    draw the all trajectory samples
    ax: matplotlib axis
    all_xs: (N, T, d) array of true state
    all_zs: (N, T, d) array of nominal state
    all_x_ests: (N, T, d) array of estimated state
    vp_est: (T, d, d) array of estimated variance proxy
    vp_tracks: (T, d, d) array of propagated variance proxy
    cp_bounds: conformal prediction bounds fromn the true state
    """
    all_xs = np.asarray(all_xs)
    all_zs = np.asarray(all_zs)
    all_x_ests = np.asarray(all_x_ests)
    draw_nominal_traj_and_samples(ax, all_xs, all_zs, all_x_ests)
    
    if cp_bounds is not None:
        length = min(len(cp_bounds), len(vp_tracks))
        for t in range(length):
            if t == 0:
                draw_ellipsoid_2d(
                    ax,
                    all_zs[0, t, 0:2],
                    cp_bounds[t][0:2, 0:2],
                    color=palettes.tue_plot[5],
                    alpha=alpha,
                    label="Conformal Prediction",
                )
            else:
                draw_ellipsoid_2d(
                    ax,
                    all_zs[0, t, 0:2],
                    cp_bounds[t][0:2, 0:2],
                    color=palettes.tue_plot[5],
                    alpha=alpha,
                )

    # draw the propagated variance proxy
    for t in range(len(vp_tracks)):
        vp_track_true_to_nominal = extract_vp_track_down(vp_tracks[t], all_xs.shape[-1])
        if t == 0:
            draw_ellipsoid_2d(
                ax,
                all_zs[0, t, 0:2],
                vp_track_true_to_nominal[0:2, 0:2] * scale**2,
                color=palettes.tue_plot[0],
                alpha=alpha,
                label="Estimated Bounds",
            )
        else:
            draw_ellipsoid_2d(
                ax,
                all_zs[0, t, 0:2],
                vp_track_true_to_nominal[0:2, 0:2] * scale**2,
                color=palettes.tue_plot[0],
                alpha=alpha,
            )


def draw_trajectories_robust(
    ax,
    all_xs,
    all_zs,
    all_x_ests,
    poly_ests,
    poly_track_ests,
    poly_track_trues,
    cp_bounds=None,
    scale=1.0,
    alpha=0.05,
):
    """
    draw the all trajectory samples
    ax: matplotlib axis
    all_xs: (N, T, d) array of true state
    all_zs: (N, T, d) array of nominal state
    all_x_ests: (N, T, d) array of estimated state
    poly_ests: (T, poly) list of estimated error robust set
    poly_track_ests: (T, poly) array of estimation to nominal error robust set
    poly_track_true: (T, poly) array of true to nominal error robust set
    """
    all_xs = np.asarray(all_xs)
    all_zs = np.asarray(all_zs)
    all_x_ests = np.asarray(all_x_ests)

    draw_nominal_traj_and_samples(ax, all_xs, all_zs, all_x_ests)

    # draw the polygon of state estimation
    # for t in range(len(poly_track_ests)):
    #     draw_translated_poly(ax, poly_track_ests[t], all_zs[0, t, :], color='purple', alpha=alpha)
    if cp_bounds is not None:
        for t in range(len(cp_bounds)):
            if t == 0:
                draw_ellipsoid_2d(
                    ax,
                    all_zs[0, t, 0:2],
                    cp_bounds[t][0:2, 0:2],
                    color=palettes.tue_plot[5],
                    alpha=alpha,
                    label="conformal prediction",
                )
            else:
                draw_ellipsoid_2d(
                    ax,
                    all_zs[0, t, 0:2],
                    cp_bounds[t][0:2, 0:2],
                    color=palettes.tue_plot[5],
                    alpha=alpha,
                )

    # draw the propagated variance proxy
    for t in range(len(poly_track_trues)):
        if t == 0:
            draw_translated_poly_points(
                ax,
                poly_track_trues[t],
                all_zs[0, t, 0:2],
                color=palettes.tue_plot[0],
                alpha=alpha,
                label="estimated bound",
            )
        else:
            draw_translated_poly_points(
                ax,
                poly_track_trues[t],
                all_zs[0, t, 0:2],
                color=palettes.tue_plot[0],
                alpha=alpha,
            )


def draw_bars_from_dict(ax, data_dict, species, label):
    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    for attribute, measurement in data_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt="%.4f")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.set_xticks(x + width, species)
    ax.legend(loc="upper left", ncols=3)
    # .set_ylim(0, 250)


def draw_1d_trajectories(
    ax,
    all_xs,
    all_zs,
    all_x_est,
    dim,
    controller,
    color=palettes.tue_plot[5],
    alpha=0.05,
):
    """
    draw the all trajectory samples
    ax: matplotlib axis
    all_xs: (N, T, d) array of true state
    all_zs: (N, T, d) array of nominal state
    controller: mpc controller
    alpha: transparency
    """
    all_xs = np.asarray(all_xs)
    all_zs = np.asarray(all_zs)
    all_x_est = np.asarray(all_x_est)
    tightens = [
        controller.tightens[i]["polygon"] for i in range(len(controller.tightens))
    ]
    tightens = np.array(tightens).reshape((-1,))
    # get t
    t = np.arange(all_xs.shape[1]) * controller.mpc_params["optimizer"]["dt"]
    # draw the true state
    for sample in range(all_xs.shape[0]):
        if sample==0:
            ax.plot(t, all_xs[sample, :, dim], c=color, alpha=alpha, label="True States")
        else:
            ax.plot(t, all_xs[sample, :, dim], c=color, alpha=alpha)
        # ax.plot(t, all_x_est[sample, :, dim], c='green', alpha=alpha)
    # constraints
    ax.hlines(
        controller.mpc_params["env"]["constraints"]["b"],
        t[0],
        t[-1],
        color=palettes.tue_plot[2],
        alpha=1.0,
        linewidth=3,
        label="Constraints",
    )
    # draw the nominal state
    if not controller.mpc_params["approach"] == "nominal":
        ax.plot(t, all_zs[0, :, dim], c=palettes.tue_plot[1], alpha=1.0, linewidth=2, label="Nominal States")
        # draw the confidence intervals
        ax.plot(
            t,
            all_zs[0, :, dim] + tightens,
            c=palettes.tue_plot[0],
            alpha=1.0,
            linewidth=2,
            label="Confidence Bounds",
        )
        # ax.plot(
        #     t,
        #     all_zs[0, :, dim] - tightens,
        #     c=palettes.tue_plot[0],
        #     alpha=1.0,
        #     linewidth=2,
        # )
    


def draw_exponential_constraints(ax, shift, radius, range, color="g", alpha=1.0):
    xs = np.linspace(range[0], range[1], 100)

    y_pos = np.exp(-(xs - shift)) + radius
    # y_neg = -np.exp(-(xs - shift)) - radius

    ax.plot(xs, y_pos, c=color, alpha=alpha, linewidth=3)
    # ax.plot(xs, y_neg, c=color, alpha=alpha)

def draw_funnel_constraints_2d(ax, x_range, x_num, const_params, color='g', alpha=1.0):
        '''
        visualize the funnel constraint
        '''
        dx = const_params["dx"]
        dy = const_params["dy"]
        dz = const_params["dz"]
        cons = const_params["constraint"]
        shift = const_params["shift"]
        # zeta = self.mpc_params["env"]["funnel"]["zeta"]
        # always visualize 0 (original funnel)
        zeta = 0

        xs = np.linspace(x_range[0], x_range[1], x_num) # (N_x)

        y_pos_up = (np.sqrt(np.exp(-xs / dx - shift) + cons) + zeta) * dy

        y_pos_down = (-np.sqrt(np.exp(-xs / dx - shift) + cons) - zeta) * dy

        ax.plot(xs, y_pos_up, c=color, 
            alpha=alpha, linewidth=3)
        ax.plot(xs, y_pos_down, c=color, 
            alpha=alpha, linewidth=3)


# visualize a set of ellipsoids
def get_ellipsoids(mats, interval):
    '''
    Given positive definite matrices, visualize ellipsoids
    '''
    
    mesh_list = []
    for i in range(len(mats)):
        if i % interval != 0:
            continue
        mat = mats[i]
        U, S, Uh = np.linalg.svd(mat[:3, :3]) # only positions

        radius = np.sqrt(S)
        mesh = pv.ParametricEllipsoid(radius[0], radius[1], radius[2])

        mesh.points = np.dot(mesh.points, U.T)
        mesh_list.append(mesh)

    return mesh_list
