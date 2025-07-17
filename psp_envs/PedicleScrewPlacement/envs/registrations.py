# registration algorithms
# GT + icp
# ransac + icp
# fast global + icp
# cpd

import open3d as o3d
import copy
import numpy as np
from .utils import get_upper_surface
from scipy.spatial.transform import Rotation as R

'''
preprocess from open3d
'''
def preprocess_point_cloud(source, voxel_size, source_normal):
    '''
    input: 
    - point array: numpy
    - voxel_size: down sample
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(source)

    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 5
    
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    normals = np.asarray(pcd_down.normals)
    mean_normals = np.mean(normals, axis=0)
    normal_dot = np.dot(mean_normals, source_normal)
    if normal_dot<0:
        # flip normals
        pcd_down.normals = o3d.utility.Vector3dVector(-normals)

    radius_feature = voxel_size * 10
    
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh, np.mean(normals, axis=0)

'''
global and local registration based on open3d library
'''
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, voxel_size, global_result):
    distance_threshold = voxel_size * 0.4
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
'''
model based registration: direct registration between upper surface and bone model
'''
def rigid_registration(global_method, local_method, voxel_size, bone_model, traj_rot, traj_enter, scanning, rand_init_range=None):
    '''
    Input:
    - global_method: "RANSAC", "Fast" or random
    - local_method: "icp" or 
    - bone_model: the input 3D vertebra GT model (from CT, MRI, etc)
    - gs_traj: ground truth annotated trajectory
    - scanning: scanned point cloud for registration

    Output:
    - T: transformation between scanning and bone model
    - transformed_scanning
    - est_traj_rot: estimated insertion trajectory direction
    - est_traj_center: estimated insertion trajectory center
    '''
    upper_bone = get_upper_surface(bone_model)
    
    # preprocessing
    rand_normal = np.random.uniform(-1, 1, (3,))
    source_down, source_fpfh, mean_source_normal = preprocess_point_cloud(upper_bone, voxel_size, rand_normal)
    target_down, target_fpfh, mean_target_normal = preprocess_point_cloud(scanning, voxel_size, mean_source_normal)

    # global registration
    if global_method=="RANSAC":
        global_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    elif global_method=="RAND":
        global_result = o3d.pipelines.registration.RegistrationResult()
        rand_euler = np.random.uniform(-rand_init_range[0] * np.ones((3,)), rand_init_range[0] * np.ones((3,)))
        rand_t = np.random.uniform(-rand_init_range[1] * np.ones((3,)), rand_init_range[1] * np.ones((3,))).reshape((1, 3))
        rand_R = R.from_euler("XYZ", rand_euler, degrees=False).as_matrix()
        trans = np.eye(4)
        trans[:3, :3] = rand_R
        trans[:3, 3] = rand_t
        global_result.transformation = trans
    elif global_method=="FAST":
        global_result = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    else:
        print("The global registration method is not supported")

    # local registration
    if local_method=="ICP":
        refine_result = refine_registration(source_down, target_down, voxel_size, global_result)
        trans = refine_result.transformation
        transformed_upper_bone = copy.deepcopy(source_down)
        transformed_upper_bone.transform(trans)
        trans_rot_mat = trans[:3, :3]
        translation = trans[:3, 3]
        transformed_upper_bone_points = np.asarray(transformed_upper_bone.points)
    elif local_method=="CPD":
        global_trans = global_result.transformation
        global_rot = global_trans[:3, :3]
        global_t = global_trans[:3, 3].reshape((1, 3))
        trans_rot_mat, translation = cpd_registration(np.asarray(upper_bone), np.asarray(scanning), global_rot, global_t)
        transformed_upper_bone_points = (trans_rot_mat @ upper_bone.T).T + translation
        trans = np.eye(4)
        trans[:3, :3] = trans_rot_mat
        trans[:3, 3] = translation

    else:
        print("The local registration method is not supported")

    est_traj_rot = copy.deepcopy(traj_rot)
    new_rot_mat = trans_rot_mat @ est_traj_rot.as_matrix()
    est_traj_rot = R.from_matrix(new_rot_mat)
    est_traj_center = trans_rot_mat @ traj_enter + translation


    return trans, transformed_upper_bone_points, est_traj_rot, est_traj_center


def cpd_registration(source, target, init_R, init_t):
    
    reg = RigidRegistration(R=init_R, t=init_t, **{'X': target, 'Y': source})
    reg.register()
    rot = reg.R
    t = reg.t
    
    return rot, t
