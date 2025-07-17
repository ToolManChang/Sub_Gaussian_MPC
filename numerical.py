import numpy as np
import skgeom as sg
from uncertainty.variance_proxy_propagation import *
from shapely.geometry import Point, Polygon
from shapely.affinity import scale, rotate
import matplotlib.pyplot as plt
from visualize import sort_clockwise
from envs import LinearEnv


def compare_area_ratios_vp(vp_list, scale, cp_bounds):

    area_ratios = []

    for t in range(len(vp_list)):
        vp = extract_vp_track_down(vp_list[t], vp_list[t].shape[0] // 2)
        cp_bound = cp_bounds[t]

        eigvals = np.linalg.eigvals(vp)
        vp_area = np.prod(np.sqrt(eigvals)) * np.pi * scale ** 2
        cp_area = cp_bound ** 2 * np.pi
        area_ratios.append(vp_area / cp_area)

    return area_ratios


def compare_area_ratios_robust(poly_list, cp_bounds):
    area_ratios = []

    for t in range(len(poly_list)):
        poly = poly_list[t]
        cp_bound = cp_bounds[t]

        poly_area = abs(float(poly.area()))
        cp_area = cp_bound ** 2 * np.pi
        area_ratios.append(poly_area / cp_area)

    return area_ratios

def compute_outlier_ratio_vp(all_xs, all_zs, vp_list, scale, last_dim=None):
    total_sample = len(all_xs)
    T = len(all_xs[0])

    all_xs_array = np.asarray(all_xs)
    all_zs_array = np.asarray(all_zs)

    if last_dim is not None:
        all_xs_array = all_xs_array[:, :, :last_dim]
        all_zs_array = all_zs_array[:, :, :last_dim]

    length = min(len(vp_list), T)

    total_ratio = 0
    for t in range(length):
        vp = vp_list[t]
        if vp.shape[0] > all_xs_array.shape[-1]:
            vp = extract_vp_track_down(vp_list[t], all_xs[0][0].shape[-1])

        if last_dim is not None:
            vp = vp[:last_dim, :last_dim]
        cur_x = all_xs_array[:, t, :]
        cur_z = all_zs_array[:, t, :]

        err = cur_x - cur_z # (N, d)

        all_cov = err @ np.linalg.inv(vp) @ err.T / scale**2 # (N, N)

        diag = np.diagonal(all_cov)

        total_ratio += np.sum(diag > 1) / total_sample

    return total_ratio / length


def compute_max_outlier_ratio_vp(all_xs, all_zs, vp_list, scale, last_dim=None):
    total_sample = len(all_xs)
    ratios = []
    T = len(all_xs[0])

    all_xs_array = np.asarray(all_xs)
    all_zs_array = np.asarray(all_zs)

    if last_dim is not None:
        all_xs_array = all_xs_array[:, :, :last_dim]
        all_zs_array = all_zs_array[:, :, :last_dim]

    length = min(len(vp_list), T)

    # total_ratio = 0
    for t in range(length):
        vp = vp_list[t]
        if vp.shape[0] > all_xs_array.shape[-1]:
            vp = extract_vp_track_down(vp_list[t], all_xs[0][0].shape[-1])

        if last_dim is not None:
            vp = vp[:last_dim, :last_dim]
        cur_x = all_xs_array[:, t, :]
        cur_z = all_zs_array[:, t, :]

        err = cur_x - cur_z # (N, d)

        diag = []
        for i in range(err.shape[0]):
            diag.append(err[i].reshape((1, -1)) @ np.linalg.inv(vp) @ err[i].reshape((-1, 1)) / scale**2)

        # all_cov = err @ np.linalg.inv(vp) @ err.T / scale**2 # (N, N)

        diag = np.array(diag).reshape((-1,))
        ratios.append(np.sum(diag > 1) / total_sample)

    return max(ratios)


def compute_unsafe_ratio(all_xs: list, env: LinearEnv):
    total_num = len(all_xs)

    all_xs_array = np.asarray(all_xs)

    num_out = 0
    for i in range(all_xs_array.shape[0]):
        for t in range(all_xs_array.shape[1]):
            cur_x = all_xs_array[i, t, :]
            if not env.check_constraint(cur_x):
                num_out += 1
                break

    return num_out / total_num


def compute_average_fail_number(all_xs: list, env: LinearEnv):
    total_num = len(all_xs)

    all_xs_array = np.asarray(all_xs)

    num_out = 0
    for i in range(all_xs_array.shape[0]):
        for t in range(all_xs_array.shape[1]):
            cur_x = all_xs_array[i, t, :]
            if not env.check_constraint(cur_x):
                num_out += 1

    return num_out / total_num

def compute_max_fail_number(all_xs: list, env: LinearEnv):

    all_xs_array = np.asarray(all_xs)

    fail_num_list = []
    for t in range(all_xs_array.shape[1]):
        num_out = 0
        for i in range(all_xs_array.shape[0]):
            cur_x = all_xs_array[i, t, :]
            if not env.check_constraint(cur_x):
                num_out += 1
        fail_num_list.append(num_out)

    return max(fail_num_list)




def compute_outlier_ratio_robust(all_xs, all_zs, poly_list):
    total_sample = len(all_xs) 
    T = len(all_xs[0])

    all_xs_array = np.asarray(all_xs)
    all_zs_array = np.asarray(all_zs)

    total_ratio = 0
    for t in range(len(poly_list)):
        poly = poly_list[t]
        poly_s = create_shapely_polygon_from_points(poly)

        cur_x = all_xs_array[:, t, :]
        cur_z = all_zs_array[:, t, :]

        err = cur_x - cur_z # (N, d)

        num_out = 0
        for i in range(err.shape[0]):
            if not poly_s.contains(Point(err[i])):
                num_out += 1
        total_ratio += num_out / total_sample

    return total_ratio / T

def compute_max_outlier_ratio_robust(all_xs, all_zs, poly_list):
    total_samples = len(all_xs)
    ratios = []

    all_xs_array = np.asarray(all_xs)
    all_zs_array = np.asarray(all_zs)

    for t in range(len(poly_list)):
        poly = poly_list[t]
        poly_s = create_shapely_polygon_from_points(poly)

        cur_x = all_xs_array[:, t, :]
        cur_z = all_zs_array[:, t, :]

        err = cur_x - cur_z # (N, d)

        num_out = 0
        for i in range(err.shape[0]):
            if not poly_s.contains(Point(err[i])):
                num_out += 1
        ratios.append(num_out / total_samples)

    return max(ratios)


def create_shapely_ellipsoid_from_vp(vp):
    eigvals, eigvecs = np.linalg.eig(vp)

    # Rotation angle (in degrees)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Create a shapely ellipse
    ellipse_center = (0, 0)
    ellipse = Point(ellipse_center).buffer(1)  # unit circle
    ellipse = scale(ellipse, np.sqrt(eigvals[0]), np.sqrt(eigvals[1]))  # scale to desired radii
    ellipse = rotate(ellipse, angle, origin=ellipse_center)  # rotate by 
    
    return ellipse


def create_shapely_polygon_from_poly(poly):

    return Polygon(poly.coords)


def create_shapely_polygon_from_points(points):

    points = sort_clockwise(points[:, 0:2])
    return Polygon(points)


def iou_shapely(shape1, shape2):
    return shape1.intersection(shape2).area / shape1.union(shape2).area


def compute_iou_vp(vp_list, scale, cp_bounds, last_dim=None):
    ious = []

    length = min(len(vp_list), len(cp_bounds))
    for t in range(length):
        vp = extract_vp_track_down(vp_list[t], vp_list[t].shape[0] // 2)
        if last_dim is not None:
            vp = vp[:last_dim, :last_dim]

        cp_bound = cp_bounds[t]

        vp_ellipse = create_shapely_ellipsoid_from_vp(vp*scale**2)
        cp_ellipse = create_shapely_ellipsoid_from_vp(cp_bound)

        ious.append(iou_shapely(vp_ellipse, cp_ellipse))

    return ious


def compute_iou_robust(poly_list, cp_bounds):
    ious = []

    for t in range(len(poly_list)):
        poly = poly_list[t]
        poly_s = create_shapely_polygon_from_points(poly)
        cp_ellipse = create_shapely_ellipsoid_from_vp(cp_bounds[t])
        
        ious.append(iou_shapely(poly_s, cp_ellipse))

    return ious


def compute_error_size_robust(poly_list, direction):
    '''
    direction: error direction
    '''
    err_sizes = []
    for t in range(len(poly_list)):
        poly = poly_list[t]
        errs = poly @ direction.reshape((-1, 1))
        err_sizes.append(np.max(errs))

    return err_sizes

def compute_error_size_sample(all_xs, all_zs, prob, direction):
    '''
    direction: error direction
    '''
    all_xs_array = np.asarray(all_xs)
    all_zs_array = np.asarray(all_zs)
    err_sizes = []
    for t in range(all_xs_array.shape[1]):
        errs = (all_xs_array[:, t, :] - all_zs_array[:, t, :]) @ direction.reshape((-1, 1))
        errs = errs.reshape((-1,))
        # sort the norms
        sort_index = np.argsort(errs)
        
        # get the index
        n = errs.shape[0]
        m = min(np.ceil(n * (1 - prob)), n-1)

        err_sizes.append(errs[sort_index[int(m)]])

    return err_sizes

def compute_error_size_vp(vp_list, scale, direction):
    '''
    direction: error direction
    '''
    err_sizes = []
    for t in range(len(vp_list)):
        if vp_list[t].shape[0] > direction.shape[0]:
            vp = extract_vp_track_down(vp_list[t], vp_list[t].shape[0] // 2)
        else:
            vp = vp_list[t]
        errs = np.sqrt(direction.reshape((1, -1)) @ vp @ direction.reshape((-1, 1))) * scale
        err_sizes.append(errs[0, 0])

    return err_sizes


def compute_average_cost(all_xs, all_us, Q, R, goal):
    '''
    all_xs: list of np.array, each with shape (N, T, d)
    all_us: np.array with shape (N, T, m)
    '''
    all_xs_array = np.asarray(all_xs)
    all_us_array = np.asarray(all_us)
    N = all_xs_array.shape[0]

    total_cost = 0
    for t in range(all_xs_array.shape[1]):
        cur_x = all_xs_array[:, t, :]
        cur_u = all_us_array[:, t, :]
        for i in range(N):
            total_cost += np.sum(
                np.diagonal((cur_x[i] - goal).reshape((1, -1)) @ Q @ (cur_x[i] - goal).reshape((-1, 1)))
            ) + np.sum(
                np.diagonal(cur_u[i].reshape((1, -1)) @ R @ cur_u[i].reshape((-1, 1)))
            )
        

    return total_cost / N