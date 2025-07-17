import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
import time
from sklearn.decomposition import PCA
import copy

'''
compute the number of points inside the box to compute the intersection area.
@params:
- points: points input (N, 3)
- center: center of the box (x, y, z)
- d: size of the box (x, y, z)
- euler: rotation of the box, from world to box coordinate (x, y, z)
'''
def voxel_box_intersection(points, center, d, rot):
    # get rotation
    rot_mat = rot.as_matrix()

    d[0] = float(d[0])
    d[1] = float(d[1])
    d[2] = float(d[2])

    # # near points
    diff = points - center
    total_d = (d[0] / 2 + d[1] / 2 + d[2] / 2)
    near_index_x = abs(diff[:, 0]) < total_d
    if not np.any(near_index_x):
        return 0
    diff_x = diff[near_index_x, :]
    near_index_xy = abs(diff_x[:, 1]) < total_d
    if not np.any(near_index_xy):
        return 0
    diff_xy = diff_x[near_index_xy, :]
    near_index_xyz = abs(diff_xy[:, 2]) < total_d
    if not np.any(near_index_xyz):
        return 0
    diff_xyz = diff_xy[near_index_xyz, :]

    # transform the points to the box coordinate:
    trans_points = (rot_mat.T @ diff_xyz.T).T # (N, 3)

    # count points
    y_select = abs(trans_points[:, 1]) < d[1] / 2
    if np.sum(y_select)==0:
        return 0
    else:
        y_select_points = trans_points[y_select, :]
    yz_select = abs(y_select_points[:, 2]) < d[2] / 2
    if np.sum(yz_select)==0:
        return 0
    else:
        yz_select_points = y_select_points[yz_select, :]
    xyz_select = abs(yz_select_points[:, 0]) < d[0] / 2

    return np.sum(xyz_select)


def voxel_cylinder_intersection(points, center, d, rot):
    # get rotation
    rot_mat = rot.as_matrix()

    d[0] = float(d[0])
    d[1] = float(d[1])
    d[2] = float(d[2])

    diff = points - center
    total_d = (d[0] / 2 + d[1] / 2 + d[2] / 2)
    near_index_x = abs(diff[:, 0]) < total_d
    if not np.any(near_index_x):
        return 0
    diff_x = diff[near_index_x, :]
    near_index_xy = abs(diff_x[:, 1]) < total_d
    if not np.any(near_index_xy):
        return 0
    diff_xy = diff_x[near_index_xy, :]
    near_index_xyz = abs(diff_xy[:, 2]) < total_d
    if not np.any(near_index_xyz):
        return 0
    diff_xyz = diff_xy[near_index_xyz, :]

    trans_points = (rot_mat.T @ diff_xyz.T).T

    # transform the points to the box coordinate:
    # trans_points = (rot_mat.T @ (points- center).T).T # (N, 3)

    x_select = abs(trans_points[:, 0]) < d[0] / 2
    if not np.any(x_select):
        return 0
    yz_select = np.linalg.norm(trans_points[x_select, 1:3], axis=1) < d[1] / 2

    return np.sum(yz_select)


def voxel_cylinder_if_intersection(points, center, d, rot):
    # # get rotation
    # rot_mat = rot.as_matrix()
    # get rotation
    rot_mat = rot.as_matrix()

    # transform the points to the box coordinate:
    # trans_points = (rot_mat.T @ (points - center).T).T # (N, 3)
    d[0] = float(d[0])
    d[1] = float(d[1])
    d[2] = float(d[2])

    diff = points - center
    total_d = (d[0] / 2 + d[1] / 2 + d[2] / 2)
    near_index_x = abs(diff[:, 0]) < total_d
    if not np.any(near_index_x):
        return 0
    diff_x = diff[near_index_x, :]
    near_index_xy = abs(diff_x[:, 1]) < total_d
    if not np.any(near_index_xy):
        return 0
    diff_xy = diff_x[near_index_xy, :]
    near_index_xyz = abs(diff_xy[:, 2]) < total_d
    if not np.any(near_index_xyz):
        return 0
    diff_xyz = diff_xy[near_index_xyz, :]

    trans_points = (rot_mat.T @ diff_xyz.T).T


    # # transform the points to the box coordinate:
    # trans_points = (rot_mat.T @ (points- center).T).T # (N, 3)

    x_select = abs(trans_points[:, 0]) < d[0] / 2
    if not np.any(x_select):
        return 0
    yz_select = np.linalg.norm(trans_points[x_select, 1:3], axis=1) < d[1] / 2

    return np.any(yz_select)


def voxel_box_if_intersection(points, center, d, rot):
    # get rotation
    rot_mat = rot.as_matrix()

    d[0] = float(d[0])
    d[1] = float(d[1])
    d[2] = float(d[2])

    # transform the points to the box coordinate:
    # trans_points = (rot_mat.T @ (points - center).T).T # (N, 3)

    diff = points - center
    total_d = (d[0] / 2 + d[1] / 2 + d[2] / 2)
    near_index_x = abs(diff[:, 0]) < total_d
    if not np.any(near_index_x):
        return 0
    diff_x = diff[near_index_x, :]
    near_index_xy = abs(diff_x[:, 1]) < total_d
    if not np.any(near_index_xy):
        return 0
    diff_xy = diff_x[near_index_xy, :]
    near_index_xyz = abs(diff_xy[:, 2]) < total_d
    if not np.any(near_index_xyz):
        return 0
    diff_xyz = diff_xy[near_index_xyz, :]

    trans_points = (rot_mat.T @ diff_xyz.T).T

    # print(trans_points[:, 0])
    # count points
    xyz_min = np.min(np.max(abs(trans_points)/np.asarray([d[0] / 2, d[1] / 2, d[2] / 2]), axis=1))

    return xyz_min < 1



def voxel_box_intersection_fast(slice, box):
    selected = slice.select_enclosed_points(box)
    return np.sum(selected['SelectedPoints'])

'''
compute the number of points inside the box to compute the intersection area.
@params:
- points: points input (N, 3)
- center: center of the box (x, y, z)
- d: size of the box (x, y, z)
- euler: rotation of the box, from world to box coordinate (x, y, z)
'''
def voxel_box_intersection_index(points, center, d, rot):
    # get rotation
    rot_mat = rot.as_matrix()

    d[0] = float(d[0])
    d[1] = float(d[1])
    d[2] = float(d[2])

    # transform the points to the box coordinate:
    trans_points = (rot_mat.T @ (points - center).T).T # (N, 3)

    # count points
    x_select = abs(trans_points[:, 0]) < d[0] / 2
    xy_select = np.logical_and(x_select, abs(trans_points[:, 1]) < d[1] / 2)
    xyz_select = np.logical_and(xy_select, abs(trans_points[:, 2]) < d[2] / 2)

    return xyz_select

'''
compute which side the points are from the box if not contact
'''
def voxel_box_side(points, center, d, rot):
    # get rotation
    rot_mat = rot.as_matrix()
    
    # compute mean
    points_center = np.mean(points, axis=0)

    # transform the points to the box coordinate:
    points_center = (rot_mat.T @ (points_center - center).T).T # (N, 3)


    return np.asarray([points_center[0] > 0, points_center[0] < 0, 
            points_center[1] > 0, points_center[1] < 0,
            points_center[2] > 0, points_center[2] < 0], np.float32)


'''
compute which side the points are from the box if not contact
'''
def voxel_box_dist(points, center, d, rot):
    # get rotation
    rot_mat = rot.as_matrix()

    # transform the points to the box coordinate:
    diff = points - center
    total_d = (d[0] / 2 + d[1] / 2 + d[2] / 2)
    near_index_x = abs(diff[:, 0]) < total_d
    if not np.any(near_index_x):
        return 0
    diff_x = diff[near_index_x, :]
    near_index_xy = abs(diff_x[:, 1]) < total_d
    if not np.any(near_index_xy):
        return 0
    diff_xy = diff_x[near_index_xy, :]
    near_index_xyz = abs(diff_xy[:, 2]) < total_d
    if not np.any(near_index_xyz):
        return 0
    diff_xyz = diff_xy[near_index_xyz, :]

    trans_points = (rot_mat.T @ diff_xyz.T).T # (N, 3)

    # count points
    lr_min_dist = 100
    dist_up = 100
    dist_down = 100

    # row up distance
    up_index = trans_points[:, 0] > d[0] / 2
    down_index = trans_points[:, 0] < -d[0] / 2
    middle_index = np.logical_and(trans_points[:, 0] < d[0] / 2, trans_points[:, 0] > - d[0] / 2)

    if np.sum(up_index) > 0:
        row_up = trans_points[up_index, :]
        dist_up = np.min(np.linalg.norm(row_up - np.asarray((d[0] / 2, 0, 0)), axis=1))

    # row down distance
    if np.sum(down_index) > 0:
        row_down = trans_points[down_index, :]
        dist_down = np.min(np.linalg.norm(row_down - np.asarray(( -d[0] / 2, 0, 0)), axis=1))

    # row middle distance
    if np.sum(middle_index) > 0:
        row_mid = trans_points[middle_index, :]

        mid_dist = np.linalg.norm(row_mid[:, 1:2], axis=1)
        
        lr_min_dist = np.min(mid_dist)
        

    # print("dist_down", dist_down)
    # print("dist_up", dist_up)
    # print("dist_left", dist_left)
    # print("dist_right", dist_right)

    return min([lr_min_dist, dist_up, dist_down])


'''
cluster points
'''
def cluster_two_sets(points):
    kmeans = KMeans(n_clusters=2).fit(points)
    return kmeans.labels_

'''
Use PCA to extract the direction of vertebra
'''
def get_PCA_components(points):
    pca = PCA(n_components=3, svd_solver='randomized')
    pca.fit(points)
    return pca.components_


def get_upper_surface(points):
    x_list = points[:, 0].astype(int)
    y_list = points[:, 1].astype(int)
    z_list = points[:, 2].astype(int)

    yz_table = np.ones((200, 200)) * 1000

    min_y = min(y_list)
    min_z = min(z_list)

    x_sort_index = (-x_list).argsort()
    
    yz_table[np.clip(y_list[x_sort_index] - min_y, 0, 199), np.clip(z_list[x_sort_index] - min_z, 0, 199)] = x_list[x_sort_index]

    upper_points = []

    z_all_index = np.tile(np.arange(0, 200), 200)
    
    y_all_index = np.repeat(np.arange(0, 200), 200)

    yz_visited = (yz_table < 1000).reshape((-1,))

    x_final = yz_table[yz_table < 1000].reshape((-1,))
    y_final = y_all_index[yz_visited] + min_y
    z_final = z_all_index[yz_visited] + min_z
    
    upper_points = np.stack([x_final, y_final, z_final], axis=1)

    upper_points = upper_points[upper_points[:, 0] < -5]

    return upper_points


'''
search for the pedicle center
'''
def search_pedicle_region(traj_rot, traj_center, vertebra, range):
    '''
    given the trajectory, search for the pedicle center
    '''
    points_in_traj_frame = (traj_rot.as_matrix().T @ (vertebra.points - traj_center).T).T

    # print(points_in_traj_frame.shape)

    select = np.logical_and(points_in_traj_frame[:, 0]<range[1], points_in_traj_frame[:, 0]>range[0])

    searched_points = points_in_traj_frame[select, :]


    searched_radius = np.linalg.norm(searched_points[:, 1:3], axis=1)

    min_index = np.argmin(searched_radius)
    min_radius = searched_radius[min_index]

    min_x = searched_points[min_index, 0]

    pedicle_center = traj_center + traj_rot.as_matrix() @ np.asarray([min_x, 0, 0])

    # compute radius
    pedicle_circle_index = np.logical_and(points_in_traj_frame[:, 0] < min_x + 0.75, points_in_traj_frame[:, 0] > min_x - 0.75)
    all_radius = np.linalg.norm(points_in_traj_frame[:, 1:3], axis=1)
    pedicle_circle_index = np.logical_and(pedicle_circle_index, all_radius<min_radius+3)
    pedicle_circle = points_in_traj_frame[pedicle_circle_index]
    pedicle_radius = np.max(np.linalg.norm(pedicle_circle[:, 1:3], axis=1))

    return pedicle_center, pedicle_radius



def traj_distance(a1, b1, a2, b2):
    '''
    compute distance between 2 trajectories
    a: point
    b: direction
    '''
    b_cross = np.cross(b1, b2)
    d = np.dot(b_cross, (a2 - a1)) / np.linalg.norm(b_cross)
    return abs(d)


def Gertzbein_Robbins_cls(screw_rot, screw_center, screw_diameter, pedicle_center, cortical_bone):
    '''
    Gertzbein robins classification for given screw and cortical bone mesh
    - screw_rot: (3, 3) matrix
    - screw center: (3,) screw origin
    - pedicle center: (3,) estimated pedicle center
    - cortical bone: (N, 3) cortical bone points
    '''

    # get projection of pedicle on bone trajectory
    pedicle_in_screw = screw_rot.T @ (pedicle_center - screw_center)

    # get projection line
    proj_line_in_screw = copy.deepcopy(pedicle_in_screw)
    proj_line_in_screw[0] = 0

    # get points on line
    n_proj_line_in_screw = proj_line_in_screw / np.linalg.norm(proj_line_in_screw)
    cortical_bone_in_screw = (screw_rot.T @ (cortical_bone.T - screw_center.reshape(-1, 1))).T
    proj_on_line_in_screw = (cortical_bone_in_screw - pedicle_in_screw.reshape(1, -1)) @ n_proj_line_in_screw
    proj_point_in_screw = proj_on_line_in_screw.reshape(-1, 1) @ n_proj_line_in_screw.reshape(1, 3)
    distance_to_line = np.linalg.norm(cortical_bone_in_screw - pedicle_in_screw.reshape(1, -1) - proj_point_in_screw, axis=1)
    close_index = np.logical_and(distance_to_line < 1.0, proj_on_line_in_screw < 0)
    close_points_in_screw = cortical_bone_in_screw[close_index, :]

    # get furthest points distance
    # distance_to_x_axis = np.linalg.norm(close_points_in_screw[:, 1:3], axis=1)
    distance_to_pedicle = np.linalg.norm(close_points_in_screw - pedicle_in_screw.reshape(1, -1), axis=1)
    
    distances = np.abs(screw_diameter / 2 + np.linalg.norm(proj_line_in_screw) - distance_to_pedicle)
    index = np.argmin(distances)
    distance = screw_diameter / 2 + np.linalg.norm(proj_line_in_screw) - distance_to_pedicle[index]

    # classification
    if distance < 0:
        GR_class = "A"
    elif distance < 2:
        GR_class = "B"
    elif distance < 4:
        GR_class = "C"
    elif distance < 6:
        GR_class = "D"
    else:
        GR_class = "E"

    return distance, GR_class


