import pyvista as pv
from pyvista import camera
import numpy as np
import os.path
from scipy.spatial.transform import Rotation as R
import gc
import pdb
from scipy.spatial.distance import cdist
from ..utils import cluster_two_sets, get_PCA_components


'''
screw together with the camera
Attributes:
- screw mesh, voxel

- position of com
- rotation of the screw

- focal point on the screw (in the screw frame)
- distance from camera to the screw
- camera focus (camera observe to -z direction)
'''
class ScrewObserver:


    '''
    initialize
    @ param:
    - model_path: path of the mesh model for screw
    - init position (x, y, z): initial position to move the screw to
    - init rotation (X, Y, Z) in screw frame: initial Euler XYZ rotation to make length in x frame
    - observe center (x, y, z): tip position (x is the length direction) in screw frame
    - box center in screw frame
    - box size (x, y, z): size of box that approximate the screw shape
    - camera dist (d): distance from the observer to the screw.
    - camera focus (f): focal length of the camera
    - tip size: a box around the observe center representing the tip area of the screw
    - head box: center in screw frame, size
    - voxelize accuracy: number of voxels for voxelization
    '''
    def __init__(self, model_path, init_rotation, init_position,  observe_center, body_center, body_size, camera_dist, camera_focus, tip_size, 
        head_center, head_size, voxelize_accuracy=100):
        
        # read models
        self.mesh = pv.read(model_path)
        self.voxel = pv.voxelize(self.mesh, density = self.mesh.length / voxelize_accuracy, check_surface=False).extract_surface().triangulate()

        # move to origin
        self.center = np.mean(self.voxel.points, axis=0)
        self.voxel.points -= self.center
        self.center = np.mean(self.voxel.points, axis=0)

        # init rotate
        self.rot = R.from_euler('XYZ', init_rotation)
        rot_mat = self.rot.as_matrix()
        self.voxel.points = self.center + (rot_mat @ (self.voxel.points - self.center).T).T
        # set the rotation zero
        self.rot = R.from_euler('XYZ', (0, 0, 0))
        rot_mat = self.rot.as_matrix()
        
        # move to initial position
        self.voxel.points += init_position
        self.center = np.mean(self.voxel.points, axis=0)

        # box center setting
        self.body_center = body_center
        self.body_size = body_size
        self.box = pv.Cube(self.center + self.body_center, self.body_size[0], self.body_size[1], self.body_size[2])
        self.cylinder = pv.voxelize(
            pv.Cylinder(
                center=self.center + self.body_center, 
                direction=[1, 0, 0],
                radius=self.body_size[1] / 2,
                height=self.body_size[0]
            ),
            check_surface=False
        )

        # camera settings
        self.observe_center = observe_center
        self.w_observe_center = observe_center
        self.camera_dist = camera_dist
        self.camera_focus = camera_focus

        # init camera in world frame
        self.camera = pv.Camera()
        self.camera.position = tuple(self.center + rot_mat @ np.asarray(self.observe_center) + rot_mat @ np.asarray((0, 0, self.camera_dist)))
        self.camera.focal_point = tuple(self.center + rot_mat @ np.asarray(self.observe_center) + rot_mat @ np.asarray((0, 0, self.camera_focus)))
        self.camera.up = tuple(-rot_mat[:, 0])

        self.side_camera = pv.Camera()
        self.side_camera.position = tuple(self.center + rot_mat @ np.asarray(self.observe_center) + rot_mat @ np.asarray((0, self.camera_dist, 0)))
        self.side_camera.focal_point = tuple(self.center + rot_mat @ np.asarray(self.observe_center) + rot_mat @ np.asarray((0, self.camera_focus, 0)))
        self.side_camera.up = tuple(-rot_mat[:, 0])

        # init tip area
        self.tip_size = tip_size
        self.w_observe_center = self.center + rot_mat @ np.asarray(self.observe_center)
        cube_center = self.w_observe_center - np.asarray((self.tip_size[0] / 2, 0, 0))
        self.tip_cube = pv.Cube(cube_center, self.tip_size[0], self.tip_size[1], self.tip_size[2])
        self.w_body_center = self.center + rot_mat @ np.asarray(self.body_center)

        # init head area
        self.head_center = head_center
        self.head_size = head_size
        self.w_head_center = self.center + rot_mat @ np.asarray(self.head_center)
        self.head_cube = pv.Cube(self.center + self.head_center, self.head_size[0], self.head_size[1], self.head_size[2])

        # coordinate frame
        self.x_axis = pv.Line(self.w_observe_center, self.w_observe_center + self.rot.as_matrix()[:,0].T * 20)
        self.y_axis = pv.Line(self.w_observe_center, self.w_observe_center + self.rot.as_matrix()[:,1].T * 20)
        self.z_axis = pv.Line(self.w_observe_center, self.w_observe_center + self.rot.as_matrix()[:,2].T * 20)

        # cutting plane
        self.cutting_plane = pv.Plane(center=self.w_observe_center, direction=self.rot.as_matrix()[:,2], i_size=30, j_size=30, i_resolution=20, j_resolution=20)




    '''
    translate the voxel according to the translation in the screw frame
    translation: (x, y, z)
    '''
    def translate_in_body_frame(self, translation):
        translation = np.asarray(translation)
        rot_mat = self.rot.as_matrix()
        # move the screw
        self.voxel.points += rot_mat @ translation
        # recompute the center
        self.center += rot_mat @ translation
        # move the camera
        self.camera.position += rot_mat @ translation
        self.camera.focal_point += rot_mat @ translation
        self.side_camera.position += rot_mat @ translation
        self.side_camera.focal_point += rot_mat @ translation
        # new box
        self.box.points += rot_mat @ translation
        self.w_body_center += rot_mat @ translation
        self.cylinder.points += rot_mat @ translation
        # compute the new cube
        self.tip_cube.points += rot_mat @ translation
        self.head_cube.points += rot_mat @ translation
        self.w_head_center += rot_mat @ translation
        self.w_observe_center += rot_mat @ translation

        self.x_axis.points[0] = self.w_observe_center
        self.x_axis.points[1] = self.w_observe_center + self.rot.as_matrix()[:,0].T * 20
        self.y_axis.points[0] = self.w_observe_center
        self.y_axis.points[1] = self.w_observe_center + self.rot.as_matrix()[:,1].T * 20
        self.z_axis.points[0] = self.w_observe_center
        self.z_axis.points[1] = self.w_observe_center + self.rot.as_matrix()[:,2].T * 20

        self.cutting_plane.points += rot_mat @ translation

        
    '''
    scale in each dimension
    '''
    def scale_yz(self, y_scale, z_scale):

        rot_mat = self.rot.as_matrix()

        points_screw_frame = (rot_mat.T @ (self.voxel.points - self.w_body_center).T).T

        points_screw_frame *= np.asarray([1.0, y_scale, z_scale])

        self.voxel.points = (rot_mat @ points_screw_frame.T).T + self.w_body_center


        c_points_screw_frame = (rot_mat.T @ (self.cylinder.points - self.w_body_center).T).T

        c_points_screw_frame *= np.asarray([1.0, y_scale, z_scale])

        self.cylinder.points = (rot_mat @ c_points_screw_frame.T).T + self.w_body_center

        # self.voxel.scale(np.asarray([1.0, y_scale, z_scale]), inplace=True)

        self.head_size *= np.asarray([1.0, y_scale, z_scale])
        self.body_size *= np.asarray([1.0, y_scale, z_scale])
        self.tip_size *= np.asarray([1.0, y_scale, z_scale])
    
    '''
    rotate the shape around the observation center
    '''
    def rotate_around_observe_center(self, XYZ):
        # get rotation change
        d_rot = R.from_euler("YZX", [XYZ[1], XYZ[2], XYZ[0]])
        d_rot_mat = d_rot.as_matrix()

        # apply rotation change
        rot_mat = self.rot.as_matrix()
        new_rot_mat = d_rot_mat @ rot_mat
        self.rot = R.from_matrix(new_rot_mat)

        # recompute center and points
        # self.w_observe_center = self.center + rot_mat @ np.asarray(self.observe_center)
        # voxel points
        self.voxel.points = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.voxel.points - self.w_observe_center).T).T
        # box 
        self.box.points = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.box.points - self.w_observe_center).T).T
        self.cylinder.points = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.cylinder.points - self.w_observe_center).T).T
        self.w_body_center = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.w_body_center - self.w_observe_center).T).T
        # tip cube
        self.tip_cube.points = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.tip_cube.points - self.w_observe_center).T).T
        # head cube
        self.head_cube.points = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.head_cube.points - self.w_observe_center).T).T
        self.w_head_center = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.w_head_center - self.w_observe_center).T).T
        # center of voxel
        self.center = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.center - self.w_observe_center).T).T
        # camera
        self.camera.up = tuple((self.center - self.w_observe_center) / np.linalg.norm(self.center - self.w_observe_center))
        self.camera.position = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.camera.position - self.w_observe_center).T).T
        self.camera.focal_point = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.camera.focal_point - self.w_observe_center).T).T
        self.side_camera.up = tuple((self.center - self.w_observe_center) / np.linalg.norm(self.center - self.w_observe_center))
        self.side_camera.position = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.side_camera.position - self.w_observe_center).T).T
        self.side_camera.focal_point = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.side_camera.focal_point - self.w_observe_center).T).T

        self.x_axis.points[0] = self.w_observe_center
        self.x_axis.points[1] = self.w_observe_center + self.rot.as_matrix()[:,0].T * 20
        self.y_axis.points[0] = self.w_observe_center
        self.y_axis.points[1] = self.w_observe_center + self.rot.as_matrix()[:,1].T * 20
        self.z_axis.points[0] = self.w_observe_center
        self.z_axis.points[1] = self.w_observe_center + self.rot.as_matrix()[:,2].T * 20

        self.cutting_plane.points = self.w_observe_center + (new_rot_mat @ rot_mat.T @ (self.cutting_plane.points - self.w_observe_center).T).T
        



class RealVertebraData:

    '''
    vertebra ground truth stl and scanned point cloud
    '''
    def __init__(self, folder, vertebra_file, point_cloud_file) -> None:

        self.gt_model = pv.read(folder + '/' + vertebra_file + '.stl')
        self.reconstruction = pv.read(folder + '/' + point_cloud_file + '.stl')

        self.visualize()

        pass

    def visualize(self):
        p = pv.Plotter()
        p.add_mesh(self.gt_model)
        p.add_mesh(self.reconstruction, color="green")
        p.show()
    


class RealSpineData:

    '''
    spine ground truth stl and scanned point cloud
    '''

    def __init__(self, root_folder, model_name, vertebra_list, point_cloud, crop_list, rot_list, trans_list_adj, rot_list_adj) -> None:
        '''
        crop_list: list of list of crop size of 2 directions
        '''

        self.gt_model_list = []
        self.gt_voxel_list = []
        self.rec_list = []
        self.total_reconstruction = pv.read(root_folder + '/' + point_cloud + '.stl')
        self.GS_right_list = []
        self.GS_left_list = []

        for i in range(len(vertebra_list)):
            cur_model = pv.read(root_folder + '/' + model_name + "_" + vertebra_list[i] + '_in_robot.stl').extract_surface().triangulate()
            # cur_voxel = pv.voxelize(cur_model, density = 1)
            # cur_voxel.save(root_folder + '/' + model_name + "_" + vertebra_list[i] + "_voxel.vtk")
            cur_voxel = pv.read(root_folder + '/' + model_name + "_" + vertebra_list[i] + '_voxel.vtk')
            center = np.mean(cur_voxel.points, axis=0)

            # got the human trajectory
            GS_right_traj = pv.read(root_folder + '/' + "Drill Canal " + vertebra_list[i] + ' right_robot.stl').extract_surface().triangulate()
            GS_left_traj = pv.read(root_folder + '/' + "Drill Canal " + vertebra_list[i] + ' left_robot.stl').extract_surface().triangulate()
        

            # compute crop vector
            pca_vectors = get_PCA_components(cur_voxel.points)
            crop_vector = pca_vectors[-1]

            v = int(vertebra_list[i][-1]) - 1

            reconstruction = self.total_reconstruction.clip(crop_vector, center, value=crop_list[v][0])
            reconstruction = reconstruction.clip(-crop_vector, center, value=crop_list[v][1])

            # move to center:
            cur_voxel.points -= center
            cur_model.points -= center
            reconstruction.points -= center
            GS_right_traj.points -= center
            GS_left_traj.points -= center

            # rotate
            rot_max = R.from_euler("XYZ", rot_list[v]).as_matrix()
            cur_voxel.points = (rot_max @ cur_voxel.points.T).T
            cur_model.points = (rot_max @ cur_model.points.T).T
            reconstruction.points = (rot_max @ reconstruction.points.T).T
            GS_right_traj.points = (rot_max @ GS_right_traj.points.T).T
            GS_left_traj.points = (rot_max @ GS_left_traj.points.T).T


            # rotate_adj
            rot_adj = R.from_euler("XYZ", rot_list_adj[v]).as_matrix()
            cur_voxel.points = (rot_adj @ cur_voxel.points.T).T
            cur_model.points = (rot_adj @ cur_model.points.T).T
            GS_right_traj.points = (rot_adj @ GS_right_traj.points.T).T
            GS_left_traj.points = (rot_adj @ GS_left_traj.points.T).T


            # translate
            cur_voxel.points += np.asarray(trans_list_adj[v])
            cur_model.points += np.asarray(trans_list_adj[v])
            GS_right_traj.points += np.asarray(trans_list_adj[v])
            GS_left_traj.points += np.asarray(trans_list_adj[v])


            # record
            self.rec_list.append(reconstruction)
            self.gt_voxel_list.append(cur_voxel)
            self.gt_model_list.append(cur_model)
            self.GS_right_list.append(GS_right_traj)
            self.GS_left_list.append(GS_left_traj)

            # print(center - self.total_reconstruction.center)
        # self.visualize()

        pass

    def visualize(self):
        p = pv.Plotter()
        p.add_mesh(self.total_reconstruction, color="green")
        p.show()
        for i in range(len(self.rec_list)):
            p = pv.Plotter()
            p.set_background("white")
            p.add_mesh(self.gt_model_list[i])
            p.add_mesh(self.rec_list[i], color="yellow")
            p.add_mesh(self.GS_right_list[i], color="green")
            p.show_axes()
            p.show()
        


