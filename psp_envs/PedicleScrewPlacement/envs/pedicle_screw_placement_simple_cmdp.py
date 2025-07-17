import pdb
from PedicleScrewPlacement.envs.objects.objects import ScrewObserver, RealSpineData
import gym
import numpy as np
from PedicleScrewPlacement.envs.utils import voxel_cylinder_if_intersection, voxel_cylinder_intersection, get_PCA_components, search_pedicle_region, traj_distance
from PedicleScrewPlacement.envs.utils import voxel_box_intersection, voxel_box_if_intersection, voxel_box_side, Gertzbein_Robbins_cls, get_upper_surface
from PedicleScrewPlacement.envs.state_estimation import posterior_estimation
import pyvista as pv
import time
from matplotlib import pyplot as plt
from ruamel.yaml import YAML
import copy
import copy
from PIL import Image
from scipy.spatial.transform import Rotation as R
import PedicleScrewPlacement.envs.obs_aug as obs_aug


# 2D environment, which have different human models to sample from
class PedicleScrewPlacementSimpleCMDP(gym.Env):

    def __init__(self, cfg_file, cfg=None):
        '''
        - cfg_file: file name
        - cfg: yaml cfg object, if specified, it will be used
        '''

        if cfg is None:
            self.cfg = YAML().load(open(cfg_file, 'r'))
        else:
            self.cfg = cfg
        
        screwFolder = self.cfg["screw_folder"]

        self.visualize_full = self.cfg["visualize_full"]

        # load vertebra
        self.anatomies = []
        self.screws = []
        self.screw_diameters = self.cfg["screw_diameters"]

        print("init_env")
        # anatomies
        self.spineData = RealSpineData(
            root_folder=self.cfg["model_folder"],
            model_name=self.cfg["model_name"],
            vertebra_list=self.cfg["vertebra_list"],
            point_cloud=self.cfg["point_cloud"],
            crop_list=self.cfg["crop_list"],
            rot_list=self.cfg["rot_list"],
            trans_list_adj=self.cfg["trans_list_adj"],
            rot_list_adj=self.cfg["rot_list_adj"],
        )
        
        # screws
        for i in range(len(self.cfg["screw"]["type"])):
        
            # read screw
            screw = ScrewObserver(screwFolder + self.cfg["screw"]["type"][i], 
                init_rotation=self.cfg["screw"]["init_rotation"][i], 
                init_position=np.asarray(self.cfg["screw"]["init_position"]), 
                observe_center=self.cfg["screw"]["observe_center"][i], 
                body_center=self.cfg["screw"]["body_center"][i],
                body_size=self.cfg["screw"]["body_size"][i], 
                camera_dist=self.cfg["screw"]["camera_dist"], 
                camera_focus=self.cfg["screw"]["camera_focus"],
                tip_size=self.cfg["screw"]["tip_size"],
                head_center=self.cfg["screw"]["head_center"][i],
                head_size=self.cfg["screw"]["head_size"])

            self.screws.append(screw)

        self.side = self.cfg["side"]

        # current selected anatomy for simulation
        self.cur_mesh = self.spineData.gt_model_list[0]
        self.cur_anatomy = self.spineData.gt_voxel_list[0]
        self.cur_screw = self.screws[0]
        self.rec_mesh = self.spineData.rec_list[0]
        if self.cfg["if_validation"]:
            self.world_upper_points = self.rec_mesh.points
        else:
            self.world_upper_points = get_upper_surface(self.cur_anatomy.points)
        if self.side=="right":
            self.GS_traj = self.spineData.GS_right_list[0]
        else:
            self.GS_traj = self.spineData.GS_left_list[0]
        
        # observations and action space
        self.obs_dim = self.cfg["obs_shape"]
        self.action_space = gym.spaces.Discrete(11) # x +-1, y +- 1, z +-1, az +- 1, ay +-1, ax +-1
        
        if self.cfg["vector_state"]:
            self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(1, 6), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=3,
                shape=(1, self.cfg["3D_observation"]["size"][0], 
                self.cfg["3D_observation"]["size"][1], 
                self.cfg["3D_observation"]["size"][2]), dtype=np.uint8)
            self.voxel_obs = np.zeros((self.cfg["3D_observation"]["size"][0], 
                self.cfg["3D_observation"]["size"][1], 
                self.cfg["3D_observation"]["size"][2]),
                dtype=np.uint8)
            

            # grid world for computing intersection faster
            self.grid = pv.UniformGrid()
            self.grid.dimensions = self.voxel_obs.shape
            self.grid.origin = (0, 0, 0)  # The bottom left corner of the data set
            self.grid.spacing = (1, 1, 1)
                

        # record overlap area (to compute rewards)
        self.body_cancellous = 0 # 0-250
        self.body_bone_side = []

        self.tip_restricted = 0
        self.tip_cortical = 0
        self.tip_cancellous = 0

        self.head_cortical = 0
        self.head_bone_side = []

        # distance from lower half to the corticle bone
        self.dist_to_cortical = 0
        self.dist_to_no_go = 0
        self.first = 1
        self.num_step = 0
        self.sub_num_step = 0

        # init plotter
        self.p = pv.Plotter(window_size=[self.cfg["window_size"][0], self.cfg["window_size"][1]], off_screen=False)
        self.obs_p = pv.Plotter(window_size=[self.cfg["window_size"][0], self.cfg["window_size"][1]], off_screen=False)
        self.p.set_background([1.0, 1.0, 1.0])
        self.obs_p.set_background([1.0, 1.0, 1.0])

        # record total cost, reward
        self.done = False
        self.last_total_cost = 0
        self.last_total_objective = 0
        self.last_total_body_restricted = 0
        self.last_total_body_bone = 0
        self.last_total_body_cortical = 0
        self.last_total_pedicle_bonus = 0
        self.last_total_dist_no_go = 0
        self.last_obs_cancellous = 0
        self.last_normal_dot = 0

        self.total_cost = 0
        self.total_objective = 0
        self.total_reward = 0
        self.total_body_restricted = 0
        self.total_body_bone = 0
        self.total_body_cortical = 0
        self.total_pedicle_bonus = 0
        self.total_dist_no_go = 0
        self.total_obs_cancellous = 0
        self.total_normal_dot = 0

        self.vertebra_side_axis = []
        self.changed_side = 0

        # modeling
        env_cfg = self.cfg
        self.A = np.array(env_cfg['A'])
        self.B = np.array(env_cfg['B'])
        self.C = np.array(env_cfg['C'])
        self.Q = np.diag(env_cfg['Q'])
        self.R = np.diag(env_cfg['R'])
        self.u_max = env_cfg['u_max']
        self.u_min = env_cfg['u_min']
        self.x_max = env_cfg['x_max']
        self.x_min = env_cfg['x_min']
        self.constraint_cfg = env_cfg['constraints']
        self.dt = env_cfg['dt']
        self.name = self.cfg['name']

        self.sample_count = 0

        print("init_env finish")


    def get_funnel_constraints(self, x_range, x_num, angle_num, const_params):
        '''
        visualize the funnel constraint
        '''
        dx = const_params["funnel"]["dx"]
        dy = const_params["funnel"]["dy"]
        dz = const_params["funnel"]["dz"]
        cons = const_params["funnel"]["constraint"]
        shift = const_params["funnel"]["shift"]
        # zeta = self.mpc_params["env"]["funnel"]["zeta"]
        # always visualize 0 (original funnel)
        zeta = 0

        xs = np.linspace(x_range[0], x_range[1], x_num) # (N_x)

        rs = np.sqrt(np.exp(-xs / dx - shift) + cons) + zeta

        all_ps = []
        for i in range(x_num):
            r = rs[i]
            x = xs[i] * np.ones((angle_num,)) # (N_angle)

            # get all angles
            angles = np.linspace(0, 2 * np.pi, angle_num)

            # get circles
            n_z = r * np.cos(angles) # (N_angle)
            n_y = r * np.sin(angles) # (N_angle)

            y = n_y * dy # (N_angle)
            z = n_z * dz # (N_angle)

            ps = np.stack([x, y, z], axis=1) # (N_angle, 3)
            all_ps.append(ps)

        all_ps = np.concatenate(all_ps, axis=0) # (N_x * N_angle, 3)
        funnel = pv.PolyData(all_ps)

        return funnel
    
    def check_constraint(self, x):
        funnel_params = self.cfg["constraints"]["funnel"]
        dx = funnel_params["dx"]
        dy = funnel_params["dy"]
        dz = funnel_params["dz"]
        cons = funnel_params["constraint"]
        shift = funnel_params["shift"]
        constraint_1 = ((x[2] / dz) ** 2
            + (x[1] / dy) ** 2
            - np.exp((-x[0] / (dx)) - shift)
            - cons)
        
        a = self.cfg['constraints']['a']
        b = self.cfg['constraints']['b']
        constrain_2 = np.dot(a, x) - b
        return constraint_1 < 0 and constrain_2 < 0
        

    
    def get_voxel_obs_coordinates(self, voxel):
        '''
        get point coordinates in the screw frame
        - voxel: pyvista PolyData object
        ''' 
        x, y, z = self.get_point_obs_coordinates(voxel.points)
        
        return x, y, z


    def get_point_obs_coordinates(self, input_points):
        '''
        Given the point cloud, compute the image coordinates in the 3D image (observation)
        '''

        # subsample
        indexes = np.arange(0, input_points.shape[0])
        np.random.shuffle(indexes) 
        indexes = indexes[0:int(input_points.shape[0] / self.cfg["3D_observation"]["subsample"])]
        
        self.obs_points = input_points[indexes, :]

        # add noise
        noise_range = np.asarray(self.cfg["3D_observation"]["noise"])
        noise = np.random.uniform(-noise_range, noise_range, self.obs_points.shape)
        self.obs_points += noise
        
        rot_mat = self.cur_screw.rot.as_matrix()
        org = self.cur_screw.w_observe_center
        pos_in_screw_frame = (rot_mat.T @ (self.obs_points - org).T).T


        # update occupancy map
        x_size = self.cfg["3D_observation"]["size"][0]
        y_size = self.cfg["3D_observation"]["size"][1]
        z_size = self.cfg["3D_observation"]["size"][2]
        x_range = [- x_size / 2 * self.cfg["3D_observation"]["density"], x_size / 2 * self.cfg["3D_observation"]["density"]]
        y_range = [- y_size / 2 * self.cfg["3D_observation"]["density"], y_size / 2 * self.cfg["3D_observation"]["density"]]
        z_range = [- z_size / 2 * self.cfg["3D_observation"]["density"], z_size / 2 * self.cfg["3D_observation"]["density"]]

        # round and select points inside the volumn
        x_list = np.round(pos_in_screw_frame[:, 0])
        y_list = np.round(pos_in_screw_frame[:, 1])
        z_list = np.round(pos_in_screw_frame[:, 2])


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
        x_list /= self.cfg["3D_observation"]["density"]
        y_list /= self.cfg["3D_observation"]["density"]
        z_list /= self.cfg["3D_observation"]["density"]

        # move to make index positive
        x_list += x_size / 2
        y_list += y_size / 2
        z_list += z_size / 2

        return x_list.astype(np.int), y_list.astype(np.int), z_list.astype(np.int)
    

    def augment_upper_surface(self):
        '''
        use augmentation functions defined in obs_aug to augment the upper surface
        '''
        size_upper_points = self.world_upper_points.shape[0]
        left_size_miss = 0
        left_size_add = 0
        total_num_point_change = 0
        
        while left_size_miss < size_upper_points * self.cfg["observation_augmentation"]["regional_missing"]["min_keep_ratio"] or left_size_add > size_upper_points * self.cfg["observation_augmentation"]["regional_addition"]["max_add_ratio"] or total_num_point_change > size_upper_points * self.cfg["observation_augmentation"]["regional_height_change"]["max_ratio"]:
            size_upper_points = self.world_upper_points.shape[0] 

            aug_world_upper_points = copy.deepcopy(self.world_upper_points)
            # # heights
            aug_world_upper_points, total_num_point_change = obs_aug.regional_heights_change(
                point_cloud=aug_world_upper_points,
                height_range=self.cfg["observation_augmentation"]["regional_height_change"]["height_range"],
                range_num_regions=self.cfg["observation_augmentation"]["regional_height_change"]["num_regions_range"],
                range_num_normals=self.cfg["observation_augmentation"]["regional_height_change"]["num_normal_range"],
                range_normal=self.cfg["observation_augmentation"]["regional_height_change"]["normal_range"]
            )

            # # addition
            aug_world_upper_points = obs_aug.regional_addition(
                point_cloud=aug_world_upper_points,
                translation_range=self.cfg["observation_augmentation"]["regional_addition"]["trans_range"],
                rotation_range=self.cfg["observation_augmentation"]["regional_addition"]["rot_range"],
                scale_range=self.cfg["observation_augmentation"]["regional_addition"]["scale_range"],
                range_num_regions=self.cfg["observation_augmentation"]["regional_addition"]["num_regions_range"],
                range_num_normals=self.cfg["observation_augmentation"]["regional_addition"]["num_normal_range"],
                range_normal=self.cfg["observation_augmentation"]["regional_addition"]["normal_range"]
            )

            left_size_add = aug_world_upper_points.shape[0]

            # # missing#
            aug_world_upper_points = obs_aug.regional_missing(
                point_cloud=aug_world_upper_points,
                range_num_regions=self.cfg["observation_augmentation"]["regional_missing"]["num_regions_range"],
                range_num_normals=self.cfg["observation_augmentation"]["regional_missing"]["num_normal_range"],
                range_normal=self.cfg["observation_augmentation"]["regional_missing"]["normal_range"]
            )
            left_size_miss = aug_world_upper_points.shape[0]

        self.world_upper_points = aug_world_upper_points

    
    def update_grid_for_voxel(self, voxel, label, voxel_obs):
        '''
        update 3D image
        - voxel: the polydata object specified to update the label in the observation (3D image)
        - label: assigned label value in the observation
        - voxel_obs: the 3D image to update
        '''
        # clear
        voxel_obs[voxel_obs==int(label)] = 0

        x_list, y_list, z_list = self.get_voxel_obs_coordinates(voxel)

        voxel_obs[(x_list).astype(np.int), (y_list).astype(np.int), (z_list).astype(np.int)] = int(label)


    def update_grid_voxel_boundary(self, voxel, label, voxel_obs):
        '''
        compute upper boundary in screw frame, and update the observation
        - voxel: object to extract upper surface to update the voxel
        - label: label in the 3D image
        - voxel_obs: 3D image to update
        '''
        voxel_obs[voxel_obs==int(label)] = 0

        x_list, y_list, z_list = self.get_point_obs_coordinates(voxel)

        voxel_obs[x_list.astype(np.int), y_list.astype(np.int), z_list.astype(np.int)] = int(label)

        
    def init_voxel_obs(self):
        '''
        init observation
        '''
        self.update_grid_voxel_boundary(self.world_upper_points, 3, self.voxel_obs)
        self.update_grid_for_voxel(self.cur_screw.voxel, 1, self.voxel_obs)
    

    def update_voxel_obs(self): 
        '''
        update 3D voxelized observation for both screw and bone
        screw not change, only update bones
        '''
        # TODO: careful! before the screw is below
        self.update_grid_for_voxel(self.cur_screw.voxel, 1, self.voxel_obs)
        self.update_grid_voxel_boundary(self.world_upper_points, 3, self.voxel_obs)


    def update_voxel_visualization(self):
        '''
        update visualization of occupancy map
        '''
        # specify the visualization values
        self.grid.point_data["values"] = self.voxel_obs.flatten(order="F") * 70
        self.grid.point_data["values"][self.grid.point_data["values"]==0] = 10
        old_actor = self.voxel_actor
        opacity_array = np.asarray([0.02, 3, 3])
        self.voxel_actor = self.obs_p.add_volume(self.grid,
            opacity=opacity_array,
            cmap = ["yellow", "blue", "green", "green", "green"],
            show_scalar_bar=False
            )
        self.obs_p.remove_actor(old_actor)
        self.obs_p.update(stime=10000)


    def adjust_human_traj(self):
        '''
        adjust human traj to be inside the bone
        '''
        GS_entry = self.get_cancellous_entry_point(self.right_traj_rot, self.GS_traj.center)
        GS_exit = self.get_cancellous_exit_point(self.right_traj_rot, self.GS_traj.center)

        self.GS_traj = pv.Cylinder(
            0.5 * (GS_entry + GS_exit),
            self.right_traj_rot.as_matrix()[:, 0],
            self.screw_diameter / 2,
            height = np.linalg.norm(GS_entry - GS_exit)
        )

        # if GS_traj is available, get the origin for control
        self.control_origin = self.GS_traj.center - self.right_traj_rot.as_matrix()[:, 0] * 100


    def validate_human_traj(self):
        '''
        compute properties related to the human traj
        '''
        
        self.right_traj_len, self.right_traj_rot, self.human_bone_penetration, self.human_unsafe = self.validate_cylinder_traj(
            self.GS_traj,
            self.screw_diameter
        )
        self.pedicle_center = np.asarray(self.GS_traj.center)
        self.adjust_human_traj()
        self.right_traj_len, self.right_traj_rot, self.human_bone_penetration, self.human_unsafe = self.validate_cylinder_traj(
            self.GS_traj,
            self.screw_diameter
        )
        

    def validate_cylinder_traj(self, cylinder, diameter):
        '''
        given the cylinder trajectory, compute the penetration value, rotation matrix and damage to restricted region
        input:
        - cylinder: cylinder object
        return:
        - rotation
        - penetration
        - if break through
        '''
        # get cylinder rotation
        right_pca = get_PCA_components(cylinder.points)
        vec_0 = right_pca[0] / np.linalg.norm(right_pca[0])
        if vec_0[0] < 0:
            vec_0 = - vec_0
        vec_1 = np.cross(np.asarray([0, 0, 1]), vec_0)
        vec_2 = np.cross(vec_0, vec_1)
        stack_pca = np.column_stack([vec_0, vec_1, vec_2])
        # make direction downwards
        # get rotation
        rot = R.from_matrix(stack_pca)
        # print(stack_pca)
    
        traj_points_in_frame = rot.as_matrix().T @ (cylinder.points - cylinder.center).T
        traj_points_in_frame = traj_points_in_frame.T
        traj_len = np.max(traj_points_in_frame[:, 0]) - np.min(traj_points_in_frame[:, 0])
        d = [traj_len, diameter, diameter]
        bone_penetration = voxel_cylinder_intersection(self.cur_anatomy.points,
            cylinder.center,
            d,
            rot
        )

        # compute exit 2 points
        exit_point = cylinder.center + rot.as_matrix()[:, 0] * traj_len / 2
        exit_point_2 = cylinder.center - rot.as_matrix()[:, 0] * traj_len / 2
        if exit_point[0] < exit_point_2[0]:
            exit_point = exit_point_2

        exit_cancellous = voxel_box_intersection(
            self.cur_anatomy.points,
            exit_point,
            self.cur_screw.tip_size,
            rot
        )


        exit_cortical = voxel_box_intersection(
            self.cur_anatomy.points,
            exit_point,
            self.cur_screw.tip_size,
            rot
        )

        break_through = bone_penetration > 0 and exit_cancellous==0 and exit_cortical==0

        return traj_len, rot, bone_penetration, break_through
    



     
    def get_cancellous_entry_point(self, rot, center):
        '''
        get cancellous entry point based on the rotation matrix and center
        Here the entry point is computed as the intersection between x-axis and the bone
        - rot: scipy R object
        - center: origin of the frame
        '''
        # screw entry point
        rot_mat = rot.as_matrix()
        org = center
        cancellous_in_screw_frame = (rot_mat.T @ (self.cur_anatomy.points - org).T).T
        # add cortical
        # cortical_in_screw_frame = (rot_mat.T @ (self.cur_anatomy.cortical_slice.points - org).T).T
        # cancellous_in_screw_frame = np.concatenate([cancellous_in_screw_frame, cortical_in_screw_frame], axis=0)

        cancellous_along_x = np.max(abs(cancellous_in_screw_frame[:, 1:3]), axis=1) < 0.6
        total_index = np.arange(0, cancellous_along_x.shape[0])
        select_along_x = total_index[cancellous_along_x]

        if np.sum(cancellous_along_x)==0:
            return np.zeros(3,)
        else:
            entry_point_index = np.argmin(cancellous_in_screw_frame[cancellous_along_x, 0])
            entry_point_screw = self.cur_anatomy.points[select_along_x[entry_point_index], :]
            return entry_point_screw
        
    '''
    get cancellous entry point
    '''
    def get_cancellous_exit_point(self, rot, center):
        '''
        Get cancellous exit point based on the rotation matrix and center
        Here the entry point is computed as the intersection between x-axis and the bone
        - rot: scipy R object
        - center: origin of the frame
        '''

        # screw entry point
        rot_mat = rot.as_matrix()
        org = center
        cancellous_in_screw_frame = (rot_mat.T @ (self.cur_anatomy.points - org).T).T
        # add cortical

        cancellous_along_x = np.max(abs(cancellous_in_screw_frame[:, 1:3]), axis=1) < 0.6
        total_index = np.arange(0, cancellous_along_x.shape[0])
        select_along_x = total_index[cancellous_along_x]

        if np.sum(cancellous_along_x)==0:
            return np.zeros(3,)
        else:
            entry_point_index = np.argmax(cancellous_in_screw_frame[cancellous_along_x, 0])
            entry_point_screw = self.cur_anatomy.points[select_along_x[entry_point_index], :]
            return entry_point_screw
            
    def compare_with_human_fast(self, center, rot):
        '''
        compute difference of trajectory between policy and human
        Inputs:
        - rot: Any rotation (scipy R object) with x-axis align with the trajectory
        - center: a point that the trajectory pass
        Outputs:
        - Angle difference
        - trajectory distance
        '''
        # distance of direction
        screw_x = rot.as_matrix()[:, 0]
        cylinder_x = self.right_traj_rot.as_matrix()[:, 0]
        drct_dist = np.arccos(np.clip(np.max(np.asarray([screw_x @ cylinder_x, - screw_x @ cylinder_x])), -1, 1))
        
        org = center
        org_cylinder = self.GS_traj.center
        # trajectory distance
        traj_dist = traj_distance(org, rot.as_matrix()[:, 0], org_cylinder, self.right_traj_rot.as_matrix()[:, 0])

        return drct_dist, traj_dist

    
    def compare_with_human(self, center, rot):
        '''
        compute difference of trajectory between policy and human
        Inputs:
        - rot: Any rotation (scipy R object) with x-axis align with the trajectory
        - center: a point that the trajectory pass
        Outputs:
        - Angle difference
        - Entry point distance
        - trajectory distance
        - width outside the bone
        '''
        # distance of direction
        screw_x = rot.as_matrix()[:, 0]
        cylinder_x = self.right_traj_rot.as_matrix()[:, 0]
        drct_dist = np.arccos(np.clip(np.max(np.asarray([screw_x @ cylinder_x, - screw_x @ cylinder_x])), -1, 1))
        # print(drct_dist, [screw_x @ cylinder_x, - screw_x @ cylinder_x])

        # distance of entry points
        # screw entry point
        org = center
        entry_point_screw = self.get_cancellous_entry_point(rot, org)

        org_cylinder = self.GS_traj.center
        entry_point_traj = self.get_cancellous_entry_point(self.right_traj_rot, org_cylinder)

        entry_point_traj_in_screw_frame = rot.as_matrix().T @ (entry_point_traj - org)

        # compute pedicle center
        self.pedicle_center, radius = search_pedicle_region(self.right_traj_rot, org_cylinder, self.cur_mesh, [-20, 20])
        pedicle_center_in_screw_frame = rot.as_matrix().T @ (self.pedicle_center.reshape((3,1)) - org.reshape((3,1)))

        # trajectory distance
        traj_dist = traj_distance(org, rot.as_matrix()[:, 0], org_cylinder, self.right_traj_rot.as_matrix()[:, 0])

        # GR evaluation
        pedicle_dist = np.linalg.norm(pedicle_center_in_screw_frame[1:3])
        GR_dist, GR_class = Gertzbein_Robbins_cls(
            screw_rot=rot.as_matrix(),
            screw_center=center,
            screw_diameter=self.screw_diameter,
            pedicle_center=self.pedicle_center,
            cortical_bone=self.cur_anatomy.points,
        )

        return drct_dist, np.linalg.norm(entry_point_traj_in_screw_frame[1:3]), traj_dist, GR_dist

    
    def update_vertebra_to_drill(self):
        '''
        compute the pose of drill frame in the vertebra frame
        output:
        - x, y, z translation
        - quaternion x, y, z, w
        '''
        pos = self.cur_screw.w_observe_center
        quat = self.cur_screw.rot.as_quat()

        self.vector_state = np.concatenate([pos, quat])





    def reset(self, seed=0):

        e = np.random.randint(0, len(self.cfg["vertebra_list"]))
        s = np.random.randint(0, len(self.cfg["screw"]["type"]))

        # force to use the longest one
        s = 0

        self.e = e

        self.cur_vertebra = self.cfg["vertebra_list"][e]
        self.cur_screw = self.screws[s]
        self.cur_anatomy = self.spineData.gt_voxel_list[e]
        self.cur_mesh = self.spineData.gt_model_list[e]
        self.rec_mesh = self.spineData.rec_list[e]
        if self.cfg["if_validation"]:
            self.world_upper_points = self.spineData.rec_list[e].points
        else:
            self.world_upper_points = get_upper_surface(self.cur_anatomy.points)
        if self.side=="right":
            self.GS_traj = self.spineData.GS_right_list[e]
        else:
            self.GS_traj = self.spineData.GS_left_list[e]
    

        if self.visualize_full:
            self.entry_point_screw = np.zeros((3,))
            self.entry_point_traj = np.zeros((3,))
            self.pedicle_center = np.zeros((3,))
        self.first = 1

        self.screw_diameter = max([float(self.screw_diameters[e]) * 0.7, 3.0])

        scale = self.screw_diameter / np.asarray(self.cur_screw.body_size)

        self.cur_screw.scale_yz(scale[1], scale[2])

        self.cur_screw.tip_size[0] = max([2.0, self.screw_diameter * 0.5])
        self.cur_screw.tip_size[1] = max([2.0, self.screw_diameter * 0.5])
        self.cur_screw.tip_size[2] = max([2.0, self.screw_diameter * 0.5])

        # augment point clouds
        if self.cfg["observation_augmentation"]["if_aug"]:
            self.augment_upper_surface()

        self.init_screw_position()

        # validate human
        self.validate_human_traj()
        shallow_traj = pv.Cylinder(self.GS_traj.center, self.right_traj_rot.as_matrix()[:, 0], self.screw_diameter / 2, self.right_traj_len)
        GS_traj = pv.voxelize(shallow_traj, check_surface=False)
        self.GS_list = []
        self.GS_list.append(GS_traj)

        # compute initial overlap area
        self.update_intersection()

        self.human_drct_diff, self.human_entry_diff, self.human_pedicle_diff, self.traj_GR = self.compare_with_human(self.cur_screw.w_body_center, self.cur_screw.rot)

        if not self.cfg["vector_state"]:
            self.init_voxel_obs()
        self.update_vertebra_to_drill()

        # init state estimator
        if self.cfg["if_state_estimation"]:
            self.prior_mu = np.zeros((6,))
            self.prior_std = np.asarray(self.cfg["state_estimator"]["prior_std"])
            self.se_ball = pv.ParametricEllipsoid(
                3 * self.prior_std[0],
                3 * self.prior_std[1],
                3 * self.prior_std[2],
            )
            self.se_ball.points = self.se_ball.points + self.prior_mu[0:3]
            self.se_control_origin_ball = pv.ParametricEllipsoid(
                3 * self.prior_std[0],
                3 * self.prior_std[1],
                3 * self.prior_std[2],
            )
            self.se_control_origin_ball.points = self.se_control_origin_ball.points + self.prior_mu[0:3] + self.control_origin

        self.funnel_constraints = self.get_funnel_constraints(
            x_range=[-0.1, 0.2],
            x_num=100,
            angle_num=100,
            const_params=self.cfg['constraints']
        )
        
        # init plotter
        self.init_plotter()

        # reset observation
        if self.cfg["vector_state"]:
            state = self.vector_state
        else:
            state = self.voxel_obs.reshape((1, self.voxel_obs.shape[0], self.voxel_obs.shape[1], self.voxel_obs.shape[2]))

        # get gt upper surface
        self.gt_upper_surface = get_upper_surface(self.cur_anatomy.points)

        # update control state
        self.update_control_state()

        # step
        self.num_step = 0

        # reset total values
        self.done = False
        self.total_cost = 0
        self.total_objective = 0
        self.total_body_bone = 0
        self.total_body_cortical = 0
        self.total_reward = 0
        self.cost_done = 0

        self.screw_traj_list = []

        return state
        

    def update_control_state(self):

        if self.cfg["visualize_full"]:
            old_actor = self.obs_actor
            self.obs_points_poly = pv.PolyData(self.obs_points)
            self.obs_actor = self.p.add_mesh(
                self.obs_points_poly,
                color=[175, 225, 175],
                opacity=0.3
            )
            self.p.remove_actor(old_actor)

        if self.cfg["if_state_estimation"]:
            mu, std = posterior_estimation(
                self.obs_points,
                self.gt_upper_surface,
                self.prior_mu, 
                self.prior_std,
                self.cfg["state_estimator"],
                cur_screw=self.cur_screw,
                obs_cfg=self.cfg["3D_observation"]
            )

            # print("mu", self.prior_mu)
            # print("std", self.prior_std)

            self.prior_mu = mu
            self.prior_std = std

            rot_mat_w_to_est_w = R.from_euler("YZX", self.prior_mu[3:]).as_matrix()

            # screw
            self.est_w_observe_center = self.cur_screw.w_observe_center
            self.est_screw_rot_mat = self.cur_screw.rot.as_matrix()

            # traj
            self.est_control_origin = rot_mat_w_to_est_w @ self.control_origin + self.prior_mu[0:3]
            self.est_traj_rot_mat = rot_mat_w_to_est_w @ self.right_traj_rot.as_matrix()

            if self.cfg["visualize_full"]:
                old_actor = self.est_actor
                self.se_ball = pv.ParametricEllipsoid(
                    max(3 * self.prior_std[0], 2),
                    max(3 * self.prior_std[1], 2),
                    max(3 * self.prior_std[2], 2),
                )
                self.se_ball.points = (rot_mat_w_to_est_w @ self.se_ball.points.T).T + self.prior_mu[0:3]
                self.est_actor = self.p.add_mesh(
                    self.se_ball,
                    color=[204, 204, 255],
                    opacity=0.5
                )
                self.p.remove_actor(old_actor)

                old_actor = self.est_control_actor
                self.se_control_origin_ball = pv.ParametricEllipsoid(
                    max(3 * self.prior_std[0], 2),
                    max(3 * self.prior_std[1], 2),
                    max(3 * self.prior_std[2], 2),
                )
                self.se_control_origin_ball.points = (rot_mat_w_to_est_w @ (self.se_control_origin_ball.points + self.control_origin).T).T + self.prior_mu[0:3]
                self.est_control_actor = self.p.add_mesh(
                    self.se_control_origin_ball,
                    color=[150, 222, 209],
                    opacity=0.5
                )
                self.p.remove_actor(old_actor)


        else: # ground truth
            # screw
            self.est_w_observe_center = self.cur_screw.w_observe_center
            self.est_screw_rot_mat = self.cur_screw.rot.as_matrix()

            # traj
            self.est_control_origin = copy.deepcopy(self.control_origin)
            self.est_traj_rot_mat = self.right_traj_rot.as_matrix()


        # get vector state in the control frame:
        # position:
        # pos = self.est_w_observe_center
        # self.pos_in_control = self.est_traj_rot_mat.T @ (pos - self.est_control_origin)
        # # orientation:
        # screw_drct = self.est_screw_rot_mat[:, 0]
        # self.angle_to_x = np.arccos(screw_drct @ self.est_traj_rot_mat[:, 0])
        # self.angle_to_yz = np.arctan2(screw_drct @ self.est_traj_rot_mat[:, 2], screw_drct @ self.est_traj_rot_mat[:, 1])
        # self.control_state = np.asarray([
        #     self.pos_in_control[0],
        #     self.pos_in_control[1],
        #     self.pos_in_control[2],
        #     self.angle_to_x,
        #     self.angle_to_yz,
        # ])
        self.control_state = self.screw_traj_poses_to_control_state(
            screw_center=self.est_w_observe_center,
            screw_rot=self.est_screw_rot_mat,
            control_origin=copy.deepcopy(self.control_origin),
            control_rot=self.est_traj_rot_mat
        )
        self.gt_control_state = self.screw_traj_poses_to_control_state(
            screw_center=self.cur_screw.w_observe_center,
            screw_rot=self.cur_screw.rot.as_matrix(),
            control_origin=self.control_origin,
            control_rot=self.right_traj_rot.as_matrix()
        )


    def screw_traj_poses_to_control_state(self, screw_center, screw_rot, control_origin, control_rot):
        '''
        compute control state based on screw poses
        '''
        # screw
        screw_drct = screw_rot[:, 0]
        angle_to_x = np.arccos(screw_drct @ control_rot[:, 0])
        angle_to_yz = np.arctan2(screw_drct @ control_rot[:, 2], screw_drct @ control_rot[:, 1])
        pos_in_control = control_rot.T @ (screw_center - control_origin)

        return np.asarray([
            pos_in_control[0],
            pos_in_control[1],
            pos_in_control[2],
            angle_to_x,
            angle_to_yz,
        ])
    

    def init_plotter(self):
        
        # print(self.visualize_full)
        # # define camera
        if self.visualize_full:
            # plotter
            self.p.clear()

            # reset camera
            self.img_camera = pv.Camera()
            self.img_camera.position = (0, 0, 900)
            self.img_camera.focal_point = (0, 0, 0)
            self.img_camera.up = (-1, 0, 0)
            self.p.camera = self.img_camera

            # boxes
            self.control_origin_ball = pv.Sphere(
                radius=5,
                center=self.control_origin
            )
            self.obs_points_poly = pv.PolyData(self.obs_points)
            self.p.add_mesh(self.cur_screw.voxel, color = 'black')
            self.obs_actor = self.p.add_mesh(self.obs_points_poly, color = [175, 225, 175], opacity=0.3)
            self.p.add_mesh(self.cur_anatomy, color='black', opacity=0.2)
            # anatomy_copy = copy.deepcopy(self.cur_anatomy)
            # anatomy_copy.points += np.ones((3,)) * 20
            # self.p.add_mesh(anatomy_copy, color='black', opacity=0.3)
            self.p.add_mesh(self.GS_traj, color='g', opacity=0.8)
            self.p.add_mesh(self.control_origin_ball, color=[100, 49, 137], opacity=0.8)
            self.p.show_axes()
            origin_ball = pv.Sphere(
                radius=5,
            )
            
            self.p.add_mesh(origin_ball, color = [9, 121, 105], opacity=0.5)

            if self.cfg["if_state_estimation"]:
                self.est_actor = self.p.add_mesh(self.se_ball, color = [204, 204, 255], opacity=0.5)
                self.est_control_actor = self.p.add_mesh(self.se_control_origin_ball, color = [150, 222, 209], opacity=0.5)

            # restricrted area
            fake_restricted = pv.Sphere(radius=self.cfg["restricted_radius"])
            # self.p.add_mesh(fake_restricted, color='pink', opacity=0.1)

            if self.cfg["if_model_based"]:
                self.p.add_mesh(self.est_right_traj, color = 'yellow', opacity = 0.6)
                self.p.add_mesh(self.trans_reconstruction, color = 'blue', opacity = 0.4)

            self.p.show(interactive_update=True)

            # 3d observation
            if not self.cfg["vector_state"]:
                self.obs_p.clear()
                self.grid.point_data["values"] = self.voxel_obs.flatten(order="F") * 70
                self.grid.point_data["values"][self.grid.point_data["values"]==0] = 10
                opacity_array = np.asarray([0.02, 3, 3])
                self.voxel_actor = self.obs_p.add_volume(self.grid,
                    opacity=opacity_array,
                    cmap = ["yellow", "blue", "green", "green", "green"],
                    show_scalar_bar=False
                    )
                self.obs_p.show(interactive=False, interactive_update=True)

            # visualize constraints
            traj_rot = self.right_traj_rot.as_matrix()
            self.funnel_constraints.points *= 1000
            self.funnel_constraints.points = self.funnel_constraints.points @ traj_rot.T + self.control_origin
            self.p.add_mesh(self.funnel_constraints, color=[135, 206, 235], opacity=0.1)


    def init_screw_position(self):
        # randomize screw pose
        self.body_no_go = 1
        self.head_no_go = 1
        done = True
        while(done):
            # sample new pose
            
            x = np.random.uniform(self.cfg["reset"]["x_range"][0], self.cfg["reset"]["x_range"][1])
            y = np.random.uniform(self.cfg["reset"]["y_range"][0], self.cfg["reset"]["y_range"][1])
            
            z = np.random.uniform(self.cfg["reset"]["z_range"][0], self.cfg["reset"]["z_range"][1])
            az = np.random.uniform(self.cfg["reset"]["z_angle_range"][0], self.cfg["reset"]["z_angle_range"][1])
            ay = np.random.uniform(self.cfg["reset"]["y_angle_range"][0], self.cfg["reset"]["y_angle_range"][1])
            ax = np.random.uniform(self.cfg["reset"]["x_angle_range"][0], self.cfg["reset"]["x_angle_range"][1])
            center = self.cur_screw.center

            rot_mat_sp = R.from_euler("YZX", [ay, az, ax]).as_matrix()
            diff_rot = self.cur_screw.rot.as_matrix().T @ rot_mat_sp
            diff_angles = R.from_matrix(diff_rot).as_euler('YZX')

            # print(ax, ay, az)

            translation = self.cur_screw.rot.as_matrix().T @ np.asarray((x - center[0], y - center[1], z - center[2]))
            self.cur_screw.translate_in_body_frame((translation[0], translation[1], translation[2]))
            self.cur_screw.rotate_around_observe_center((diff_angles[2], diff_angles[0], diff_angles[1]))

            # update intersection
            self.update_intersection()

            # filter the init position
            done = False
            # if self.head_cortical:
            #     done = True
            # motion range
            if self.cur_screw.rot.as_euler("YZX")[2] < self.cfg["reset"]["x_angle_range"][0]:
                done = True
            if self.cur_screw.rot.as_euler("YZX")[2] > self.cfg["reset"]["x_angle_range"][1]:
                done = True
            if self.cur_screw.rot.as_euler("YZX")[0] < self.cfg["reset"]["y_angle_range"][0]:
                done = True
            if self.cur_screw.rot.as_euler("YZX")[0] > self.cfg["reset"]["y_angle_range"][1]:
                done = True
            if self.cur_screw.rot.as_euler("YZX")[1] < self.cfg["reset"]["z_angle_range"][0]:
                done = True
            if self.cur_screw.rot.as_euler("YZX")[1] > self.cfg["reset"]["z_angle_range"][1]:
                done = True
            
            if self.cur_screw.center[0] < self.cfg["reset"]["x_range"][0]:
                done = True
            if self.cur_screw.center[0] > self.cfg["reset"]["x_range"][1]:
                done = True
            if self.cur_screw.center[1] < self.cfg["reset"]["y_range"][0]:
                done = True
            if self.cur_screw.center[1] > self.cfg["reset"]["y_range"][1]:
                done = True
            if self.cur_screw.center[2] < self.cfg["reset"]["z_range"][0]:
                done = True
            if self.cur_screw.center[2] > self.cfg["reset"]["z_range"][1]:
                done = True

    
    def update_intersection(self):
        '''
        update current intersection area
        '''

        # self.body_cancellous = voxel_cylinder_intersection(self.cur_anatomy.points, 
        #         self.cur_screw.w_body_center, 
        #         self.cur_screw.body_size, 
        #         self.cur_screw.rot)
        self.bone_cancellous = 0


        self.body_bone_side = voxel_box_side(self.cur_anatomy.points, 
                self.cur_screw.w_body_center, 
                self.cur_screw.body_size, 
                self.cur_screw.rot)


        self.head_bone_side = voxel_box_side(self.cur_anatomy.points, 
                self.cur_screw.w_head_center, 
                self.cur_screw.head_size, 
                self.cur_screw.rot)


        # self.tip_restricted = np.linalg.norm(self.cur_screw.w_observe_center - np.mean(self.cur_anatomy.points)) < self.cfg["restricted_radius"]
        self.tip_restricted = 0

        # self.tip_cancellous = voxel_box_if_intersection(self.cur_anatomy.points, 
        #         self.cur_screw.w_observe_center, 
        #         self.cur_screw.tip_size, 
        #         self.cur_screw.rot)
        self.tip_cancellous = 0

        # self.head_cortical = voxel_cylinder_if_intersection(self.cur_anatomy.points, 
        #         self.cur_screw.w_head_center, 
        #         self.cur_screw.head_size, 
        #         self.cur_screw.rot)
        self.head_cortical = 0

        
    def move(self, dx, dy, dz, dax, day, daz):   
        '''
        move the screw according to a reasonable dynamics
        - dx, dy, dz: x, y, z translation in the screw frame
        - dax, day, daz: X, Y, Z euler angle
        '''

        # filter according to head
        if self.head_cortical>0:
            if self.head_bone_side[0] > 0.5:
                dx = min(dx, 0)
            if self.head_bone_side[1] > 0.5:
                dx = max(dx, 0)
            if self.head_bone_side[2] > 0.5:
                dy = min(dy, 0)
                daz = min(daz, 0)
            if self.head_bone_side[3] > 0.5:
                dy = max(dy, 0)
                daz = max(daz, 0)
            if self.head_bone_side[4] > 0.5:
                dz = min(dz, 0)
                day = min(day, 0)
            if self.head_bone_side[5] > 0.5:
                dz = max(dz, 0)
                day = max(day, 0)

        # outside the body
        if self.tip_restricted==0 and self.body_cancellous==0:
            self.cur_screw.translate_in_body_frame((dx, dy, dz))
            self.cur_screw.rotate_around_observe_center((dax, day, daz))

        # part inside soft anatomy, not in the bone
        if self.tip_restricted > 0 and self.body_cancellous==0:
            self.cur_screw.translate_in_body_frame((self.cfg["motion_ratio"]["restricted"][0] * dx, self.cfg["motion_ratio"]["restricted"][1] * dy, self.cfg["motion_ratio"]["restricted"][2] * dz))
            self.cur_screw.rotate_around_observe_center((dax, self.cfg["motion_ratio"]["restricted"][3] * day, self.cfg["motion_ratio"]["restricted"][4] * daz))

        # break through
        if self.tip_cancellous==0 and self.body_cancellous>0:
            self.cur_screw.translate_in_body_frame((self.cfg["motion_ratio"]["bone"][0] * dx, self.cfg["motion_ratio"]["bone"][1] * dy, self.cfg["motion_ratio"]["bone"][2] * dz))
            # self.cur_screw.rotate_around_observe_center((dax, self.cfg["motion_ratio"]["bone"][1] * day, self.cfg["motion_ratio"]["bone"][2] * daz))


        # part inside the cancellous, not break through
        if self.tip_cancellous>0:
            self.cur_screw.translate_in_body_frame((self.cfg["motion_ratio"]["bone"][0] * dx, self.cfg["motion_ratio"]["bone"][1] * dy, self.cfg["motion_ratio"]["bone"][2] * dz))
            # self.cur_screw.rotate_around_observe_center((dax, self.cfg["motion_ratio"]["bone"][1] * day, self.cfg["motion_ratio"]["bone"][2] * daz))


    '''
    gaussian motion noise
    '''
    def add_driller_motion_noise(self):

        # normal motion noise, rotation and translation
        n = self.cfg["motion_noise"]["pose_laplace"]
        u = self.cfg["motion_noise"]["uniform"]
        b = self.cfg["motion_noise"]["bound"]

        translation = np.random.laplace(0, n, (3,))
        translation[np.abs(translation) > b] = b
        translation += np.random.uniform(-u, u, (3,))
        dx = translation[0]
        dy = translation[1]
        dz = translation[2]

        rotation = np.random.laplace(0, n * 0.01, (3,))
        rotation[np.abs(rotation) > b * 0.01] = b
        rotation += np.random.uniform(-u, u, (3,)) * 0.01
        dax = rotation[0] * 0.1
        day = rotation[1]
        daz = rotation[2]
      
        # outside the body
        if self.tip_restricted==0 and self.body_cancellous==0 or not self.cfg['motion_ratio']['if_constraint']:
            self.cur_screw.translate_in_body_frame((dx, dy, dz))
            self.cur_screw.rotate_around_observe_center((dax, day, daz))

        # part inside soft anatomy, not in the bone
        if self.tip_restricted > 0 and self.body_cancellous==0:
            self.cur_screw.translate_in_body_frame((self.cfg["motion_ratio"]["restricted"][0] * dx, self.cfg["motion_ratio"]["restricted"][1] * dy, self.cfg["motion_ratio"]["restricted"][2] * dz))
            self.cur_screw.rotate_around_observe_center((dax, self.cfg["motion_ratio"]["restricted"][3] * day, self.cfg["motion_ratio"]["restricted"][4] * daz))

        # break through
        if self.tip_cancellous==0 and self.body_cancellous>0:
            self.cur_screw.translate_in_body_frame((self.cfg["motion_ratio"]["bone"][0] * dx, self.cfg["motion_ratio"]["bone"][1] * dy, self.cfg["motion_ratio"]["bone"][2] * dz))
            self.cur_screw.rotate_around_observe_center((dax, self.cfg["motion_ratio"]["bone"][1] * day, self.cfg["motion_ratio"]["bone"][2] * daz))


        # part inside the cancellous, not break through
        if self.tip_cancellous>0:
            self.cur_screw.translate_in_body_frame((self.cfg["motion_ratio"]["bone"][0] * dx, self.cfg["motion_ratio"]["bone"][1] * dy, self.cfg["motion_ratio"]["bone"][2] * dz))
            self.cur_screw.rotate_around_observe_center((dax, self.cfg["motion_ratio"]["bone"][1] * day, self.cfg["motion_ratio"]["bone"][2] * daz))

    
    def compute_safe_distance(self):
        '''
        compute distance to the furthest cancellous point along the x axis within a width
        '''
        rot_mat = self.cur_screw.rot.as_matrix()
        org = self.cur_screw.w_observe_center
        cancellous_in_screw_frame = (rot_mat.T @ (self.cur_anatomy.points - org).T).T

        cancellous_along_x = np.linalg.norm(cancellous_in_screw_frame[:, 1:3], axis=1) < self.cur_screw.body_size[1] / 2 * self.cfg["safe_dist_width_ratio"]
        
        if not np.any(cancellous_along_x):
            dist_cancellous = 300
        else:
            selected_cancellous = cancellous_in_screw_frame[cancellous_along_x, 0]
            int_selected_cancellous = (np.round(selected_cancellous / 2) + 150).astype(np.int) 

            x_table = np.zeros((300,))
            x_down_table = np.zeros((300,))
            int_selected_cancellous = np.clip(int_selected_cancellous, 0, 298)

            int_selected_cancellous_down = (int_selected_cancellous + 1).astype(np.int) 

            x_down_table[int_selected_cancellous_down] = 1
            x_table[int_selected_cancellous] = 1
            cancellous_lower_bound = np.logical_and(x_table == 0, x_down_table == 1)

            # if has segment
            if np.sum(cancellous_lower_bound) > 0:
                indexes = np.arange(0, 300)
                dist_cancellous = min(indexes[cancellous_lower_bound==1]) * 2 - 300.
                # print(dist_cancellous)
            else:
                dist_cancellous = max(selected_cancellous)

        return dist_cancellous
    
    def translate_screw_frame_action_to_control_frame(self, action):
        '''
        input:
        - action: a list of 5-dim motions corresponding to motion in control frames:
        x, y, z translation, angle to x-axis, angle to y-axis in yz plane
        output:
        - motion command 
        x, y, z in screw frame, dax, day, daz for new rotation
        '''
        # translations
        rot_mat_screw_to_control = self.cur_screw.rot.as_matrix().T @ self.est_traj_rot_mat # (R_SC)
        trans_in_control = np.asarray(action[0:3])
        trans_in_screw = rot_mat_screw_to_control @ trans_in_control

        # rotations
        # get new angles
        new_alpha = self.control_state[3] + action[3]
        new_beta = self.control_state[4] + action[4]

        # compute new control to screw rot
        new_rot_control_to_screw = R.from_euler("XZX", [new_beta, new_alpha, -new_beta]).as_matrix()

        # compute new world to screw rot
        new_rot_screw = self.est_traj_rot_mat @ new_rot_control_to_screw

        # compute difference of rotations
        rot_screw_new_to_old = new_rot_screw @ self.cur_screw.rot.as_matrix().T

        # convert to euler angles
        euler_action = R.from_matrix(rot_screw_new_to_old).as_euler("YZX")

        return trans_in_screw[0], trans_in_screw[1], trans_in_screw[2], euler_action[2], euler_action[0], euler_action[1]


    def step(self, action):
        '''
        from action, move the objects, get the next state
        - action: 0-10
        - 0-5: translation, 6-9: rotation, 10: still
        '''
        dx = 0
        dy = 0
        dz = 0
        dax = 0
        day = 0
        daz = 0

        # # TODO: DEBUG
        if not type(action) == list:
        
            if action==0:
                dx = 5
            if action==1:
                dx = -5
            if action==2:
                dy = 5
            if action==3:
                dy = -5
            if action==4:
                dz = 2
            if action==5:
                dz = -2
            if action==6:
                day = 0.02
            if action==7:
                day = -0.02
            if action==8:
                daz = 0.05
            if action==9:
                daz = -0.05

        else:
            dx, dy, dz, dax, day, daz = self.translate_screw_frame_action_to_control_frame(action)
            # print(dx, dy, dz, dax, day, daz)


        # save current overlap
        prev_body_cancellous = self.body_cancellous
        prev_human_drct_diff = self.human_drct_diff
        prev_human_entry_diff = self.human_entry_diff


        # move the body
        try:
            s = self.cfg["motion_range"]["scale"]
            if self.cfg["motion_ratio"]['if_constraint']:
                self.move(dx * s, dy * s, dz * s, dax * s, day * s, daz * s)
            else:
                self.cur_screw.translate_in_body_frame((dx, dy, dz))
                self.cur_screw.rotate_around_observe_center((dax, day, daz))
        except:
            print("move error")

        try:
            self.add_driller_motion_noise()
        except:
            print("diller noise error")

        try:
            self.update_intersection()
        except:
            print("update intersection error")
        
        # TODO: safe distance
        try:
            safe_dist = self.compute_safe_distance()
            safe_dist /= 300
        except:
            print("safe dist compute error")

        # get new observation
        self.update_vertebra_to_drill()
        if not self.cfg["vector_state"]:
            try:
                self.update_voxel_obs()
            except:
                print("update obs error")

        # update control state
        self.update_control_state()

        # new trajectory distance
        try:
            self.human_drct_diff, self.human_entry_diff = self.compare_with_human_fast(self.cur_screw.w_body_center, self.cur_screw.rot)
        except:
            print("compare with human error")

        
        if self.cfg["vector_state"]:
            state = self.vector_state
        else:
            state = self.voxel_obs.reshape((1, self.voxel_obs.shape[0], self.voxel_obs.shape[1], self.voxel_obs.shape[2]))

        if self.visualize_full:
            self.p.update(stime=10000)
            if not self.cfg["vector_state"]:
                self.update_voxel_visualization()
        

        # compute cost
        cost = 0.
        # time cost
        cost_done = False
        # break through
        if self.tip_cancellous==0 and self.body_cancellous>0:
            cost_done = True

        # cross central line while inserting the screw
        if self.cur_screw.w_body_center[1] * self.cur_screw.w_observe_center[1] < 0 and self.body_cancellous > 0:
            cost_done = True

        # motion range
        if self.cur_screw.rot.as_euler("YZX")[2] < self.cfg["motion_range"]["x_angle"][0]:
            cost_done = True
        if self.cur_screw.rot.as_euler("YZX")[2] > self.cfg["motion_range"]["x_angle"][1]:
            cost_done = True
        if self.cur_screw.rot.as_matrix()[1, 0] < self.cfg["motion_range"]["y_orien"][0]:
            cost_done = True
        if self.cur_screw.rot.as_matrix()[1, 0] > self.cfg["motion_range"]["y_orien"][1]:
            cost_done = True
        if self.cur_screw.rot.as_matrix()[2, 0] < self.cfg["motion_range"]["z_orien"][0]:
            cost_done = True
        if self.cur_screw.rot.as_matrix()[2, 0] > self.cfg["motion_range"]["z_orien"][1]:
            cost_done = True
        
        if self.cur_screw.w_body_center[0] < self.cfg["motion_range"]["x"][0]:
            cost_done = True
        if self.cur_screw.w_body_center[0] > self.cfg["motion_range"]["x"][1]:
            cost_done = True
        if self.cur_screw.w_body_center[1] < self.cfg["motion_range"]["y"][0]:
            cost_done = True
        if self.cur_screw.w_body_center[1] > self.cfg["motion_range"]["y"][1]:
            cost_done = True
        if self.cur_screw.w_body_center[2] < self.cfg["motion_range"]["z"][0]:
            cost_done = True
        if self.cur_screw.w_body_center[2] > self.cfg["motion_range"]["z"][1]:
            cost_done = True
        
        if cost_done:
            cost = 1.

        self.cost_done = cost_done

        # reward
        reward = 0.
        # reward additional penetration
        if self.body_cancellous>0 and not ((self.tip_cancellous==0)) and not cost_done:
            if safe_dist > self.cfg["reward_safe_dist_thr"]:
                if self.side == "right" and self.cur_screw.center[1] < 0 or self.side == "left" and self.cur_screw.center[1] > 0.01:
                    reward += (self.body_cancellous - prev_body_cancellous) / self.screw_diameter**2 * 20 * self.cfg["reward_weights"]["insertion_depth"]
                   
        # penalty for unsafe behavior
        if not self.cost_done and cost_done:
            self.cost_done = 1
            reward -= max(0, self.cfg["reward_weights"]["unsafe_penalty"] * self.body_cancellous) / self.screw_diameter**2 * 20

        # follow human planning
        reward -= self.cfg["reward_weights"]["human_drct_diff"] * (self.human_drct_diff - prev_human_drct_diff)
        reward -= self.cfg["reward_weights"]["human_entry_diff"] * (self.human_entry_diff - prev_human_entry_diff)

        # penalty for damaging human tissues
        if self.tip_restricted and not self.tip_cancellous and not self.body_cancellous:
            if not day==0 or not daz==0 or not dy==0 or not dz==0:
                reward -= self.cfg["reward_weights"]["restricted"]
        
        done = False
        # TODO: no termination
        if self.num_step == self.cfg["max_steps"]-1:
            done = True

        if self.num_step >= self.cfg["max_steps"]-1 and self.cfg["visualize_history"]: 
            self.human_drct_diff, self.human_entry_diff, self.human_pedicle_diff, self.traj_GR = self.compare_with_human(self.cur_screw.w_body_center, self.cur_screw.rot)

        # visualize history
        if self.cfg["visualize_history"] and self.num_step % 5 == 0:
            tmp_screw = copy.copy(self.cur_screw.voxel)
            tmp_cylinder = copy.copy(self.cur_screw.cylinder)
            
            self.screw_traj_list.append(tmp_screw)
            self.screw_traj_list.append(tmp_cylinder)
            
            self.p.add_mesh(tmp_screw, color = 'black', opacity=0.1)

        # record values
        self.done = done
        self.total_cost += cost
        self.total_body_bone += (self.body_cancellous - prev_body_cancellous)
        

        if done:
            self.last_total_cost = self.total_cost
            self.last_total_objective = self.total_objective
            self.last_total_body_bone = self.total_body_bone
            self.last_total_body_cortical = self.total_body_cortical
            
        self.insertion_depth = self.total_body_bone / self.screw_diameter**2 * 4 / np.pi

        info = {
                "body_cancellous": self.body_cancellous, # 0-250
                "body_bone_side": self.body_bone_side,
                "tip_cancellous":self.tip_cancellous,
                "head_cortical": self.head_cortical,
                "cost": cost,
                "total_reward": self.total_reward,
                "total_cost": self.total_cost,
                "total_body_bone": self.insertion_depth,
                "bone stepwise reward": self.body_cancellous - prev_body_cancellous,
                "human_bone": self.human_bone_penetration / self.screw_diameter**2 * 4 / np.pi,
                "human_direction_diff": self.human_drct_diff,
                "human_entry_diff": self.human_pedicle_diff,
                "insertion_depth_ratio": self.total_body_bone / self.human_bone_penetration,
                "traj_GR": self.traj_GR,
                "safe_dist": safe_dist,
                "control_state": self.control_state,
                "vector_state": self.vector_state
                }
        

        total_reward = reward
        self.total_reward += total_reward
        total_reward *= self.cfg["reward_scale"]
        self.num_step += 1


        return state, total_reward, done, info
    
    
    
