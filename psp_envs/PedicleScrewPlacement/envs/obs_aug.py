import numpy as np
import copy as copy
from scipy.spatial.transform import Rotation as R



'''
data augmentation to augment the point cloud, functions include:
- gaussian noise
- regional height change
- regional missing
- regional addition
- random extrinsics error (future)
'''


def add_noise(point_xyz, variances, uniform_range):
    '''
    input:
    - point cloud: list [x_list, y_list, z_list]
    - variances: [x_sigma, y_sigma, z_sigma]
    - uniform_range: [max_x, max_y, max_z]
    '''
    x_list, y_list, z_list = point_xyz[0], point_xyz[1], point_xyz[2]

    x_noise = np.random.normal(0.0, np.ones(x_list.shape) * variances[0])
    x_uniform_noise = np.random.uniform(np.ones(x_list.shape) * (-uniform_range[0]), np.ones(x_list.shape) * uniform_range[0])
    y_noise = np.random.normal(0.0, np.ones(x_list.shape) * variances[1])
    y_uniform_noise = np.random.uniform(np.ones(x_list.shape) * (-uniform_range[1]), np.ones(x_list.shape) * uniform_range[1])
    z_noise = np.random.normal(0.0, np.ones(x_list.shape) * variances[2])
    z_uniform_noise = np.random.uniform(np.ones(x_list.shape) * (-uniform_range[2]), np.ones(x_list.shape) * uniform_range[2])

    x_list += x_noise + x_uniform_noise
    y_list += y_noise + y_uniform_noise
    z_list += z_noise + z_uniform_noise

    return [x_list, y_list, z_list]


def points_inside_polygon(point_cloud, center, normals):
    '''
    select points inside one polygon determined by center
    - point cloud: n*3 figure
    - center: center of the polygon
    - normals: normals from center to the half space that define the half spaces
    '''
    point_indexes = np.ones(point_cloud.shape[0], dtype=np.bool8)
    total_indexes = np.arange(point_cloud.shape[0])

    for normal in normals:
        inner_product = (point_cloud[point_indexes, 1:3] - center[1:3]) @ normal.reshape((2, 1))
        # remove the points that are outside the half space
        outside_halfspace = (inner_product > np.linalg.norm(normal) ** 2).reshape((-1,))
        remaining_indexes = total_indexes[point_indexes]
        point_indexes[remaining_indexes[outside_halfspace]] = 0

    return point_indexes

    

def select_regions(point_cloud, num_regions, range_num_normals, range_normal):
    '''
    select several regions in yz plane (polygons) and extract corresponding points
    the polygons are represented by a set of vectors, which are the normals of the half spaces that go through the origin 
    - point cloud: np.array (n * 3)
    - num_regions: number of local area in 2D plane to extract
    - range line: range of 2D normals that represent convex hull [min_p, max_p]
    - range_num_normal: range of number of half spaces [min_num, max_num]
    return:
    - list of bool indicator of whether the point is in the polygons
    '''

    # select the centers of the polygons
    select_center_index = np.random.choice(np.arange(point_cloud.shape[0]), num_regions)
    centers = point_cloud[select_center_index, :]


    points_in_polygons = []
    # sample normals
    for i_polygon in range(num_regions):
        num_normals = np.random.randint(range_num_normals[0], range_num_normals[1])
        normals = np.random.uniform(range_normal[0], range_normal[1], (num_normals, 2))

        # get indexes of points inside the polygons
        p_in_polygon = points_inside_polygon(point_cloud, centers[i_polygon, :], normals)

        # combine them
        points_in_polygons.append(p_in_polygon)
    
    # print(np.sum(points_in_polygons))

    return points_in_polygons


def regional_missing(point_cloud, range_num_regions, range_num_normals, range_normal):
    '''
    - point cloud: np.array (n * 3)
    - range num_regions: range of number of local area in 2D plane to extract [min, max]
    - range line: range of normals that represent convex hull [min_p, max_p]
    - range_num_normal: range of number of half spaces [min_num, max_num]
    return:
    - the point cloud that does not contain the missing parts
    '''
    num_regions = np.random.randint(range_num_regions[0], range_num_regions[1])

    points_in_regions_indexes = select_regions(
        point_cloud,
        num_regions,
        range_num_normals,
        range_normal
    )

    all_polygon_points_index = np.zeros(point_cloud.shape[0], np.bool8)

    for polygon_points in points_in_regions_indexes:
        all_polygon_points_index[polygon_points] = 1

    return point_cloud[np.logical_not(all_polygon_points_index)]


def regional_heights_change(point_cloud, height_range, range_num_regions, range_num_normals, range_normal):
    '''
    - point cloud: np.array (n * 3)
    - height_range: range of height changes for the regions
    - range num_regions: range of number of local area in 2D plane to extract [min, max]
    - range line: range of normals that represent convex hull [min_p, max_p]
    - range_num_normal: range of number of half spaces [min_num, max_num]
    return:
    - the point cloud that change the heights
    - number of points that change the heights
    '''
    num_regions = np.random.randint(range_num_regions[0], range_num_regions[1])
    total_region = 0

    points_in_regions_indexes = select_regions(
        point_cloud,
        num_regions,
        range_num_normals,
        range_normal
    )

    random_heights = np.random.uniform(np.ones(num_regions)*height_range[0], np.ones(num_regions)*height_range[1])

    point_cloud_copy = copy.deepcopy(point_cloud)

    for i in range(num_regions):
        point_cloud_copy[points_in_regions_indexes[i], 0] += random_heights[i]
        total_region += np.sum(points_in_regions_indexes[i])

    return point_cloud_copy, total_region


def regional_addition(point_cloud, translation_range, rotation_range, scale_range, range_num_regions, range_num_normals, range_normal):
    '''
    In this function, we basicly try to add some additional pieces to simulate unexpected additional points from other veterbra, 
    tissues, etc. 
    As in most cases additional points are from another veterbra, so the additional points are also selected from the current vertebra,
    Then it is copied rescaled, rotated and translated to a new position near the original point cloud
    - point cloud: np.array (n * 3)
    - translation_range: [min_xyz, max_xyz] range of translation for the regions
    - rotation_range: [min_angle, max_angle], range of angles for the regions
    - scale range: [min_scale, max_scale] range of scale for the copied pieces 
    - range num_regions: range of number of local area in 2D plane to extract [min, max]
    - range normal: range of normals that represent convex hull [min_p, max_p]
    - range_num_normal: range of number of half spaces [min_num, max_num]
    return:
    - the point cloud that with additional pieces
    '''
    # sample number of regions
    num_regions = np.random.randint(range_num_regions[0], range_num_regions[1])

    # get polygon points
    points_in_regions_indexes = select_regions(
        point_cloud,
        num_regions,
        range_num_normals,
        range_normal
    )

    # sample translations, scales and rotation for the points
    trans = np.random.uniform(
        np.ones((num_regions, 1)) @ np.asarray(translation_range[0]).reshape(1, -1), 
        np.ones((num_regions, 1)) @ np.asarray(translation_range[1]).reshape(1, -1)
    )

    scales = np.random.uniform(
        np.ones((num_regions, 1)) @ np.asarray(scale_range[0]).reshape(1, -1), 
        np.ones((num_regions, 1)) @ np.asarray(scale_range[1]).reshape(1, -1)
    )

    angles = np.random.uniform(
        np.ones((num_regions, 1)) @ np.asarray(rotation_range[0]).reshape(1, -1), 
        np.ones((num_regions, 1)) @ np.asarray(rotation_range[1]).reshape(1, -1)
    )

    # copy the points of each polygon:
    point_cloud_copy = copy.deepcopy(point_cloud)

    for i in range(num_regions):
        # copy polygon points
        add_points = copy.deepcopy(point_cloud[points_in_regions_indexes[i]])
        # translate polygon points
        add_points += trans[i, :]
        # scale
        add_points_center = np.mean(add_points, axis=0)
        add_points = add_points_center + (add_points - add_points_center) * scales[i, :]
        # rotate:
        rot = R.from_euler('XYZ', [angles[i], 0, 0])
        rot_max = rot.as_matrix()
        add_points = add_points_center + (rot_max @ (add_points - add_points_center).T).T

        # add to point cloud
        point_cloud_copy = np.concatenate([point_cloud_copy, add_points], axis=0)

    return point_cloud_copy


    

