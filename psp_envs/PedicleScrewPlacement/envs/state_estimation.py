# estimate bone pose in the world frame
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import open3d as o3d

def posterior_estimation(obs_surface, GT_surface, prior_mu, prior_std, se_config, if_RL_filter=False, cur_screw=None, obs_cfg=None, model=None):
    '''
    Given prior estimated distribution of pose, update the posterior estimation
    -num_samples: number of particles
    '''
    num_samples = se_config["num_samples"]
    likelihood = se_config["likelihood"]

    # sample points from prior
    # particles = np.random.normal(prior_mu, prior_std, (num_samples, prior_mu.shape[0]))
    particles = np.random.uniform(-prior_std, prior_std, (num_samples, prior_mu.shape[0]))

    # convert pose to point clouds
    rot_mats = [R.from_euler("YZX", particles[i, 3:]).as_matrix() for i in range(particles.shape[0])]
    trans = particles[:, 0:3]

    # transform point clouds
    transformed_pcds = [(rot_mats[i] @ (GT_surface + trans[i, :]).T).T for i in range(len(rot_mats))]

    # compute weights
    if likelihood=="voxel":
        pred_observations = [point_cloud_to_voxel_obs(transformed_surface, cur_screw, obs_cfg) for transformed_surface in transformed_pcds]
        observation = point_cloud_to_voxel_obs(obs_surface, cur_screw, obs_cfg)
        weights = [unnormalized_likelihood_function_voxels(pred_observations[i], observation) for i in range(len(pred_observations))]
    elif likelihood=="distance":
        weights = [unnormalized_likelihood_function(transformed_pcds[i], obs_surface) for i in range(len(transformed_pcds))]
    else:
        print("likehood type not supported")
    weights = np.asarray(weights)
    
    w_sum = sum(weights)
    n_weights = np.asarray(weights) / w_sum


    # update mu and std
    # post_mu = np.sum(particles * n_weights.reshape((-1, 1)), axis=0)
    post_mu = particles[np.argmax(n_weights), :]
    # post_var = np.sum(np.square(particles - post_mu) * n_weights.reshape((-1, 1)), axis=0)
    post_std = np.std(particles - post_mu, axis=0)
    # post_std = np.sqrt(post_var)
    min_std = np.asarray((0.2, 0.2, 0.2, 0.002, 0.002, 0.002))
    post_std = np.max(np.stack([post_std, min_std]), axis=0)

    return post_mu, post_std

def points_to_obs_coordinates(points, cur_screw, obs_cfg):
    # subsample
    indexes = np.arange(0, points.shape[0])
    np.random.shuffle(indexes) 
    indexes = indexes[0:int(points.shape[0] / obs_cfg["subsample"])]
    
    obs_points = points[indexes, :]

    # rotate
    rot_mat = cur_screw.rot.as_matrix()
    org = cur_screw.w_observe_center
    pos_in_screw_frame = (rot_mat.T @ (obs_points - org).T).T


    # update occupancy map
    x_size = obs_cfg["size"][0]
    y_size = obs_cfg["size"][1]
    z_size = obs_cfg["size"][2]
    x_range = [- x_size / 2 * obs_cfg["density"], x_size / 2 * obs_cfg["density"]]
    y_range = [- y_size / 2 * obs_cfg["density"], y_size / 2 * obs_cfg["density"]]
    z_range = [- z_size / 2 * obs_cfg["density"], z_size / 2 * obs_cfg["density"]]

    # round and select points inside the volumn
    x_list = np.round(pos_in_screw_frame[:, 0])
    y_list = np.round(pos_in_screw_frame[:, 1])
    z_list = np.round(pos_in_screw_frame[:, 2])

    # add constraints
    select_index = np.arange(0, x_list.shape[0])

    select_index = select_index[x_list[select_index] < x_range[1]]
    select_index = select_index[x_list[select_index] >= x_range[0]]
    select_index = select_index[y_list[select_index] < y_range[1]]
    select_index = select_index[y_list[select_index] >= y_range[0]]
    select_index = select_index[z_list[select_index] < z_range[1]]
    select_index = select_index[z_list[select_index] >= z_range[0]]

    x_list = x_list[select_index]
    y_list = y_list[select_index]
    z_list = z_list[select_index]

    # devide by the density
    x_list /= obs_cfg["density"]
    y_list /= obs_cfg["density"]
    z_list /= obs_cfg["density"]

    # move to make index positive
    x_list += x_size / 2
    y_list += y_size / 2
    z_list += z_size / 2

    return (x_list).astype(np.int), (y_list).astype(np.int), (z_list).astype(np.int)

def point_cloud_to_voxel_obs(pcd, cur_screw, obs_cfg):
    '''
    given the point cloud coordinates in world frame,
    compute the observation of the screw
    '''
    
    voxel_obs = np.zeros((obs_cfg["size"][0], 
                obs_cfg["size"][1], 
                obs_cfg["size"][2]),
                dtype=np.uint8)
    
    x_list, y_list, z_list = points_to_obs_coordinates(pcd, cur_screw, obs_cfg)
    voxel_obs[(x_list).astype(np.int), (y_list).astype(np.int), (z_list).astype(np.int)] = 3

    # points = cur_screw.voxel.points
    # x_list_screw, y_list_screw, z_list_screw = points_to_obs_coordinates(points, cur_screw, obs_cfg)
    # voxel_obs[(x_list_screw).astype(np.int), (y_list_screw).astype(np.int), (z_list_screw).astype(np.int)] = 1

    return voxel_obs.reshape((1,) + voxel_obs.shape)




def unnormalized_likelihood_function(est_upper_surface, obs_surface):
    '''
    Given the two surfaces, estimate the likelihood (distance between surface)
    '''
    est_pcd = o3d.geometry.PointCloud()
    est_pcd.points = o3d.utility.Vector3dVector(est_upper_surface)

    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(obs_surface)

    dists = est_pcd.compute_point_cloud_distance(obs_pcd)

    return np.exp(- np.mean(dists) / 100)

def unnormalized_likelihood_function_voxels(est_voxel, obs_voxel):
    '''
    Given the two surfaces, estimate the likelihood (distance between surface)
    '''
    same_pixels = np.sum(np.logical_and(est_voxel==3, obs_voxel==3))

    return np.exp(same_pixels / 100)
