import pyvista as pv
from pyvista import camera
import numpy as np
import os.path
from scipy.spatial.transform import Rotation as R
import gc
import math
import pandas as pd
from ruamel.yaml import YAML
import h5py 
import pydicom as dicom
import open3d as o3d

'''
A region around the interested vertebra
@Attributes:
bones list [cortical bone, cancellous bone], voxels
can be drilled region list, voxels
restricted region list, voxels
no_go_region list, voxels, slices
voxelization accuracies of the four regions in order 
(cortical, cancellous, can be drilled, restricted, no go)
a slice plane to take 2D representation
slice thickness to cut from the original anatomy
'''
class AnatomyRegionProcess:

    '''
    @ params:
    human: folder to take anatomies
    vertebra: L1-L5
    voxel_accuracies: voxelization accuracies of the four regions in order 
    (cortical, cancellous, can be drilled, restricted, no go)
    slice thickness: thickness to cut a slice from the region to get 2D simulation
    slice plane: a slice plance to take 2D representation
    '''
    def __init__(self, human, vertebra, save_folder, voxel_accuracies=[100, 100, 300, 200, 300], 
                label_map_to_origin_trans = [105.03999625,267.03496275,1050.35926585],
                ):

        self.human = human
        self.vertebra = vertebra
        self.voxel_accuracies = voxel_accuracies
        self.save_folder = save_folder
        self.label_map_to_origin_trans = np.asarray(label_map_to_origin_trans)

        
        # list of mesh from different categories
        self.cancellous_drilled = []
        self.cortical_drilled = []
        self.can_be_drilled = []
        self.restricted_drilled = []
        self.no_go_region = []
        self.cur_bone = []

        # TODO: construct the human planned trajectories
        self.clinic_standard_df = pd.read_csv(self.human + "clinic_standard.csv", index_col=0)
        print(self.clinic_standard_df)
        left_traj = np.asarray(self.clinic_standard_df.at[vertebra + " - left", "Trajectory"].split(",")).astype(np.float)
        right_traj = np.asarray(self.clinic_standard_df.at[vertebra + " - right", "Trajectory"].split(",")).astype(np.float)
        left_screw_diameter = self.clinic_standard_df.at[vertebra + " - left", "Screw Ø [mm]"]
        right_screw_diameter = self.clinic_standard_df.at[vertebra + " - right", "Screw Ø [mm]"]
        left_cortical_entry_point =  np.asarray(self.clinic_standard_df.at[vertebra + " - left", "Entry Point Cortical"].split(",")).astype(np.float)
        right_cortical_entry_point =  np.asarray(self.clinic_standard_df.at[vertebra + " - right", "Entry Point Cortical"].split(",")).astype(np.float)
        left_cortical_exit_point =  np.asarray(self.clinic_standard_df.at[vertebra + " - left", "Exit Point Cortical"].split(",")).astype(np.float)
        right_cortical_exit_point =  np.asarray(self.clinic_standard_df.at[vertebra + " - right", "Exit Point Cortical"].split(",")).astype(np.float)
        left_cancellous_entry_point =  np.asarray(self.clinic_standard_df.at[vertebra + " - left", "Entry Point Cancellous"].split(",")).astype(np.float)
        right_cancellous_entry_point =  np.asarray(self.clinic_standard_df.at[vertebra + " - right", "Entry Point Cancellous"].split(",")).astype(np.float)
        left_cancellous_exit_point =  np.asarray(self.clinic_standard_df.at[vertebra + " - left", "Exit Point Cancellous"].split(",")).astype(np.float)
        right_cancellous_exit_point =  np.asarray(self.clinic_standard_df.at[vertebra + " - right", "Exit Point Cancellous"].split(",")).astype(np.float)
        self.clinic_standard = {
            "left_traj": left_traj,
            "right_traj": right_traj,
            "left_screw_diameter": left_screw_diameter,
            "right_screw_diameter": right_screw_diameter,
            "left_cortical_entry_point": left_cortical_entry_point,
            "right_cortical_entry_point": right_cortical_entry_point,
            "left_cortical_exit_point": left_cortical_exit_point,
            "right_cortical_exit_point": right_cortical_exit_point,
            "left_cancellous_entry_point": left_cancellous_entry_point,
            "right_cancellous_entry_point": right_cancellous_entry_point,
            "left_cancellous_exit_point": left_cancellous_exit_point,
            "right_cancellous_exit_point": right_cancellous_exit_point
        }

        # build cylinders
        left_traj_center = (left_cancellous_exit_point + left_cancellous_entry_point) / 2
        left_traj_len = np.linalg.norm(left_cancellous_exit_point - left_cancellous_entry_point)

        self.left_traj_mesh = pv.Cylinder(
            center=left_traj_center, 
            direction=left_traj,
            radius=left_screw_diameter / 2.0,
            height=left_traj_len
        )

        
        right_traj_center = (right_cancellous_exit_point + right_cancellous_entry_point) / 2
        right_traj_len = np.linalg.norm(right_cancellous_exit_point - right_cancellous_entry_point)
        self.right_traj_mesh = pv.Cylinder(
            center=right_traj_center, 
            direction=right_traj,
            radius=right_screw_diameter / 2.0,
            height=right_traj_len
        )


        self.categorize_anatomy()

        # voxelized regions
        self.cortical_voxel = []
        self.cancellous_voxel = []
        self.can_be_drilled_voxel = []
        self.restricted_voxel = []
        self.no_go_voxel = []
        self.voxelization_and_merge()

        # read ultrasound reconstruction
        # self.read_reconstruction_h5()
        self.read_reconstruction_pcd()

        # mean points
        self.all_points = np.concatenate([self.no_go_voxel.points,
            self.restricted_voxel.points,
            self.cortical_voxel.points, 
            self.cancellous_voxel.points], axis=0)
        self.center = np.mean(self.all_points, axis=0)


        # move to origin
        self.cortical_slice = []
        self.cancellous_slice = []
        self.can_be_drilled_slice = []
        self.restricted_slice = []
        self.no_go_slice = []
        self.voxel_move_to_origin()
        self.vertebra_center = np.mean(self.cur_cancellous_voxel.points, axis=0)

        self.CT_p = pv.Plotter()
        self.CT_p.set_background("black")
        self.CT_p.add_mesh(self.no_go_voxel, color = 'w', opacity=(0. + 766.)/(3071.+ 766.)*0.1, lighting=False)
        self.CT_p.add_mesh(self.restricted_voxel, color = 'w', opacity=(0. + 766.)/(3071.+ 766.)*0.1, lighting=False)
        self.CT_p.add_mesh(self.can_be_drilled_voxel, color = 'w', opacity=(-500. + 766.)/(3071.+ 766.)*0.1, lighting=False)
        self.CT_p.add_mesh(self.cortical_voxel, color = 'w', opacity=(2000. + 766.)/(3071.+ 766.)*0.1, lighting=False)
        self.CT_p.add_mesh(self.cancellous_voxel, color="w", opacity=(500. + 766.)/(3071.+ 766.)*0.1, lighting=False)

        camera = pv.Camera()
        camera.position = (-750, 0, 0)
        camera.focal_point = (1500, 0, 0)
        camera.up = (0, 0, -1)

        self.CT_p.camera = camera

        # self.CT_p.show()
        




    '''
    select the useful anatomies and categorize them into 4 regions
    '''
    def categorize_anatomy(self):
        folderName = self.human
        vertebra = self.vertebra

        # bones desired to be dirlled
        veterbra_mesh = pv.read(folderName + "Vertebra_cortical_" + vertebra + ".stl").triangulate()
        self.cur_bone.append(veterbra_mesh)
        veterbra_mesh_can = pv.read(folderName + "Vertebra_cancellous_" + vertebra + ".stl").triangulate()
        self.cur_bone.append(veterbra_mesh_can)

        # all bones
        for cur_vertebra in ["L1", "L2", "L3", "L4", "L5"]:
            veterbra_mesh = pv.read(folderName + "Vertebra_cortical_" + cur_vertebra + ".stl").triangulate()
            self.cortical_drilled.append(veterbra_mesh)
            veterbra_mesh_can = pv.read(folderName + "Vertebra_cancellous_" + cur_vertebra + ".stl").triangulate()
            self.cancellous_drilled.append(veterbra_mesh_can)



        # can be drilled
        skin = pv.read(folderName + "Skin.stl")
        self.can_be_drilled.append(skin)
        SAT_org = pv.read(folderName + "SAT.stl")
        self.can_be_drilled.append(SAT_org)

        # as less as possible
        tendon = pv.read(folderName + "Tendon_ligament.stl")
        self.restricted_drilled.append(tendon)
        Muscle = pv.read(folderName + "Muscle.stl")
        ## for fats
        # max_p = np.max(Muscle.points, axis=0) - 2
        # min_p = np.min(Muscle.points, axis=0) + 2
        # box = pv.Box(bounds=(min_p[0], max_p[0], min_p[1], max_p[1], min_p[2], max_p[2]), quads=False)
        # Muscle = Muscle.clip_surface(box)
        # p1 = pv.Plotter()
        # p1.add_mesh(Muscle)
        # p1.show()
        self.restricted_drilled.append(Muscle)
        Fat = pv.read(folderName + "Fat.stl")
        self.restricted_drilled.append(Fat)

        # no go region
        spine_cord = pv.read(folderName + "Spinal_cord.stl")
        self.no_go_region.append(spine_cord)
        vein = pv.read(folderName + "Vein.stl")
        self.no_go_region.append(vein)
        Ureter = pv.read(folderName + "Ureter.stl")
        self.no_go_region.append(Ureter)
        pelvis = pv.read(folderName + "Pelvis_cortical.stl")
        self.no_go_region.append(pelvis)
        Nerve = pv.read(folderName + "Nerve.stl")
        self.no_go_region.append(Nerve)
        Kidney = pv.read(folderName + "Kidney_cortex.stl")
        self.no_go_region.append(Kidney)
        Intervertebral_disc = pv.read(folderName + "Intervertebral_disc.stl")
        self.no_go_region.append(Intervertebral_disc)
        Cerebrospinal_fluid = pv.read(folderName + "Cerebrospinal_fluid.stl")
        self.no_go_region.append(Cerebrospinal_fluid)
        Artery = pv.read(folderName + "Artery.stl")
        self.no_go_region.append(Artery)

        # plot to show the result
        p = pv.Plotter()
        self.CT_p = pv.Plotter()
        self.CT_p.set_background("black")
        for anatomy in self.no_go_region:
            p.add_mesh(anatomy, color="r", opacity=0.5)
            self.CT_p.add_mesh(anatomy, color = 'w', opacity=(100. + 766.)/(3071.+ 766.))
        for anatomy in self.restricted_drilled:
            p.add_mesh(anatomy, color="pink", opacity=0.4)
            self.CT_p.add_mesh(anatomy, color = 'w', opacity=(50. + 766.)/(3071.+ 766.))
        for anatomy in self.can_be_drilled:
            p.add_mesh(anatomy, color="orange", opacity=0.1)
            self.CT_p.add_mesh(anatomy, color = 'w', opacity=(-500. + 766.)/(3071.+ 766.))
        
        for anatomy in self.cancellous_drilled:
            p.add_mesh(anatomy, color="purple", opacity=0.5)
            self.CT_p.add_mesh(anatomy, color = 'w', opacity=(-500. + 766.)/(3071.+ 766.))

        # trajectories
        p.add_mesh(self.left_traj_mesh, color="g", opacity=0.8)
        p.add_mesh(self.right_traj_mesh, color="g", opacity=0.8)
        p.show()
        
        

        # camera = pv.Camera()
        # camera.position = (750, 0, 0)
        # camera.focal_point = (-1500, 0, 0)
        # camera.up = (0, 0, 1)

        # self.CT_p.camera = camera

        # self.CT_p.show()

    '''
    merge and get voxelization of four regions
    '''
    def voxelization_and_merge(self):

        # vertebra
        print("voxelize cur bone")
        if not os.path.exists(self.human + "cortical_voxel" + self.vertebra + ".vtk"):
            self.cur_cortical_voxel = pv.voxelize(self.cur_bone[0], density = self.voxel_accuracies[0])
            self.cur_cancellous_voxel = pv.voxelize(self.cur_bone[1], density = self.voxel_accuracies[1])
            self.cur_cortical_voxel.save(self.human + "cortical_voxel" + self.vertebra + ".vtk")
            self.cur_cancellous_voxel.save(self.human + "cancellous_voxel" + self.vertebra + ".vtk")
        else:
            self.cur_cortical_voxel = pv.read(self.human + "cortical_voxel" + self.vertebra + ".vtk")
            self.cur_cancellous_voxel = pv.read(self.human + "cancellous_voxel" + self.vertebra + ".vtk")
        

        print("voxelize cortical bone")
        if not os.path.exists(self.human + "cortical_voxel" + ".vtk"):
            self.cortical_voxel = pv.voxelize(self.cortical_drilled[0], density = self.voxel_accuracies[0])
            i = 0
            for anatomy in self.cortical_drilled:
                print(i, "/", len(self.cortical_drilled))
                if i == 0:
                    i += 1
                    continue
                voxels = pv.voxelize(anatomy, density = self.voxel_accuracies[0])
                self.cortical_voxel = self.cortical_voxel.merge(voxels)
                i += 1
            self.cortical_voxel.save(self.human + "cortical_voxel" + ".vtk")
        else:
            self.cortical_voxel = pv.read(self.human + "cortical_voxel" + ".vtk")
        print("cortical bone done", "density: ", self.voxel_accuracies[0])

        print("voxelize cancellous bone")
        if not os.path.exists(self.human + "cancellous_voxel" + ".vtk"):
            self.cancellous_voxel = pv.voxelize(self.cancellous_drilled[0], density = self.voxel_accuracies[1])
            i = 0
            for anatomy in self.cancellous_drilled:
                print(i, "/", len(self.cancellous_drilled))
                if i == 0:
                    i += 1
                    continue
                voxels = pv.voxelize(anatomy, density = self.voxel_accuracies[1])
                self.cancellous_voxel = self.cancellous_voxel.merge(voxels)
                i += 1
            self.cancellous_voxel.save(self.human + "cancellous_voxel" + ".vtk")
        else:
            self.cancellous_voxel = pv.read(self.human + "cancellous_voxel" + ".vtk")
        print("cancellous bone done", "density: ", self.voxel_accuracies[1])

        # can_be_drilled
        print("voxelize can be drilled region")
        if not os.path.exists(self.human + "can_be_drilled_voxel.vtk"):
            self.can_be_drilled_voxel = pv.voxelize(self.can_be_drilled[0], density = self.voxel_accuracies[2])
            i = 0
            for anatomy in self.can_be_drilled:
                print(i, "/", len(self.can_be_drilled))
                voxels = pv.voxelize(anatomy, density = self.voxel_accuracies[2])
                self.can_be_drilled_voxel = self.can_be_drilled_voxel.merge(voxels)
                i += 1

            # save for faster process
            self.can_be_drilled_voxel.save(self.human + "can_be_drilled_voxel.vtk")
        else:
            self.can_be_drilled_voxel = pv.read(self.human + "can_be_drilled_voxel.vtk")
        print("can be drilled region done", "density: ", self.voxel_accuracies[2])


        # restricted
        print("voxelize restricted region")
        if not os.path.exists(self.human + "restricted_voxel.vtk"):
            self.restricted_voxel = pv.voxelize(self.restricted_drilled[0], density = self.voxel_accuracies[3])
            i = 0
            for anatomy in self.restricted_drilled:
                print(i, "/", len(self.restricted_drilled))
                voxels = pv.voxelize(anatomy, density = self.voxel_accuracies[3], check_surface=False)
                self.restricted_voxel = self.restricted_voxel.merge(voxels)
                i += 1

            # save for faster process
            self.restricted_voxel.save(self.human + "restricted_voxel.vtk")
        else:
            self.restricted_voxel = pv.read(self.human + "restricted_voxel.vtk")
        print("restricted region done", "density: ", self.voxel_accuracies[3])

        # no go
        print("voxelize no go region")
        if not os.path.exists(self.human + "no_go_voxel.vtk"):
            self.no_go_voxel = pv.voxelize(self.no_go_region[0], density = self.voxel_accuracies[4])
            i = 0
            for anatomy in self.no_go_region:
                print(i, "/", len(self.no_go_region))
                voxels = pv.voxelize(anatomy, density = self.voxel_accuracies[4])
                self.no_go_voxel = self.no_go_voxel.merge(voxels)
                i += 1
            
            # save for faster process
            self.no_go_voxel.save(self.human + "no_go_voxel.vtk")
        else:
            self.no_go_voxel = pv.read(self.human + "no_go_voxel.vtk")

        print("no go region done", "density: ", self.voxel_accuracies[4])

        # visualize the voxel of each region
        p = pv.Plotter()
        p.add_mesh(self.no_go_voxel, color="r", opacity=0.5)
        p.add_mesh(self.restricted_voxel, color="pink", opacity=0.4)
        p.add_mesh(self.can_be_drilled_voxel, color="orange", opacity=0.1)
        p.add_mesh(self.cortical_voxel, color="w", opacity=0.5)
        p.add_mesh(self.cancellous_voxel, color="w", opacity=1)
        p.show()


    '''
    move to origin
    '''
    def voxel_move_to_origin(self):
        self.no_go_voxel.points -= self.center
        self.restricted_voxel.points -= self.center
        self.cortical_voxel.points -= self.center
        self.can_be_drilled_voxel.points -= self.center
        self.cancellous_voxel.points -= self.center

        self.cur_cortical_voxel.points -= self.center
        self.cur_cancellous_voxel.points -= self.center

        # trajectories
        self.left_traj_mesh.points -= self.center
        self.right_traj_mesh.points -= self.center

        # reconstruction
        self.point_cloud.points -= self.center

    '''
    rotation
    '''
    def voxel_rotate_y_around_vertebra(self, angle):
        axes = pv.Axes(show_actor=False, actor_scale=2.0, line_width=5)
        self.vertebra_center = np.mean(self.cur_cancellous_voxel.points, axis=0)
        axes.origin = self.vertebra_center

        self.no_go_voxel = self.no_go_voxel.rotate_y(angle, point=axes.origin, inplace=False)
        self.restricted_voxel = self.restricted_voxel.rotate_y(angle, point=axes.origin, inplace=False)
        self.can_be_drilled_voxel = self.can_be_drilled_voxel.rotate_y(angle, point=axes.origin, inplace=False)
        self.cancellous_voxel = self.cancellous_voxel.rotate_y(angle, point=axes.origin, inplace=False)
        self.cortical_voxel = self.cortical_voxel.rotate_y(angle, point=axes.origin, inplace=False)

        # trajectories
        self.left_traj_mesh = self.left_traj_mesh.rotate_y(angle, point=axes.origin, inplace=False)
        self.right_traj_mesh = self.right_traj_mesh.rotate_y(angle, point=axes.origin, inplace=False)

        # reconstruction
        self.point_cloud = self.point_cloud.rotate_y(angle, point=axes.origin, inplace=False)


    '''
    get slice for 2D representation
    '''
    def take_slice(self, slice_thickness, direction, cut_translation, visualize=True):

        self.slice_thickness = slice_thickness
        self.cut_direction = np.asarray(direction)
        self.cut_translation = cut_translation

        rot = R.from_euler("XYZ", (-math.asin(self.cut_direction[1]), math.asin(self.cut_direction[0]), 0))
        rot_mat = rot.as_matrix()

        cut_pos = self.vertebra_center + np.asarray(self.cut_translation)
        # clip regions
        
        self.no_go_slice = self.no_go_voxel.clip(self.cut_direction, -0.02 * self.cut_direction + cut_pos)
        

        
        self.restricted_slice = self.restricted_voxel.clip(self.cut_direction, -0.05 * self.cut_direction + cut_pos)
        
        
        self.cortical_slice = self.cortical_voxel.clip(self.cut_direction, 0.01 * self.cut_direction + cut_pos)
        

        
        self.can_be_drilled_slice = self.can_be_drilled_voxel.clip(self.cut_direction, -0.03 * self.cut_direction + cut_pos)
        

        
        self.cancellous_slice = self.cancellous_voxel.clip(self.cut_direction, (0,0,0) + cut_pos)

        self.point_cloud = self.point_cloud.clip(self.cut_direction, -4 * self.cut_direction + cut_pos)

        if visualize:
            p = pv.Plotter()
            p.add_mesh(self.no_go_slice, color = 'r', opacity=0.3)
            p.add_mesh(self.restricted_slice, color = 'pink',opacity=0.3)
            p.add_mesh(self.cortical_slice, color = 'black', opacity=0.5)
            p.add_mesh(self.cancellous_slice, color = 'w', lighting=1.0)
            p.add_mesh(self.can_be_drilled_slice, color = 'orange', opacity=0.1)
            p.add_mesh(self.point_cloud, color = 'y', lighting=0.7)
            p.set_background('w')
            p.show()


        
        self.no_go_slice = self.no_go_slice.clip(-self.cut_direction, -self.slice_thickness * self.cut_direction + cut_pos)
        # rotate to the z
        self.no_go_slice.points = (rot_mat.T @ self.no_go_slice.points.T).T
        self.no_go_slice.save(self.human + self.save_folder + "/no_go_slice" + "%.2f" % self.cut_translation[2] 
                              + "(%.2f," % direction[0] 
                              + "%.2f," % direction[1]
                              + "%.2f," % direction[2]
                              + ").vtk")

        
        self.restricted_slice = self.restricted_slice.clip(-self.cut_direction, -self.slice_thickness * self.cut_direction + cut_pos)
        self.restricted_slice.points = (rot_mat.T @ self.restricted_slice.points.T).T
        self.restricted_slice.save(self.human +  self.save_folder + "/restricted_slice" + "%.2f" % self.cut_translation[2] 
                              + "(%.2f," % direction[0] 
                              + "%.2f," % direction[1]
                              + "%.2f," % direction[2]
                              + ").vtk")

        
        self.cortical_slice = self.cortical_slice.clip(-self.cut_direction, -self.slice_thickness * self.cut_direction + cut_pos)
        self.cortical_slice.points = (rot_mat.T @ self.cortical_slice.points.T).T
        self.cortical_slice.save(self.human +  self.save_folder + "/cortical_slice" + "%.2f" % self.cut_translation[2] 
                              + "(%.2f," % direction[0] 
                              + "%.2f," % direction[1]
                              + "%.2f," % direction[2]
                              + ").vtk")

       
        self.can_be_drilled_slice = self.can_be_drilled_slice.clip(-self.cut_direction, -self.slice_thickness * self.cut_direction + cut_pos)
        self.can_be_drilled_slice.points = (rot_mat.T @ self.can_be_drilled_slice.points.T).T
        self.can_be_drilled_slice.save(self.human + self.save_folder+ "/can_be_drilled_slice" + "%.2f" % self.cut_translation[2] 
                              + "(%.2f," % direction[0] 
                              + "%.2f," % direction[1]
                              + "%.2f," % direction[2]
                              + ").vtk")

        
        self.cancellous_slice = self.cancellous_slice.clip(-self.cut_direction, -self.slice_thickness * self.cut_direction + cut_pos)
        self.cancellous_slice.points = (rot_mat.T @ self.cancellous_slice.points.T).T
        self.cancellous_slice.save(self.human +  self.save_folder + "/cancellous_slice" + "%.2f" % self.cut_translation[2] 
                              + "(%.2f," % direction[0] 
                              + "%.2f," % direction[1]
                              + "%.2f," % direction[2]
                              + ").vtk")

        # safe trajectories
        self.left_traj_mesh.points = (rot_mat.T @ self.left_traj_mesh.points.T).T
        self.left_traj_mesh.save(self.human +  self.save_folder + "/left_traj" + "%.2f" % self.cut_translation[2] 
                              + "(%.2f," % direction[0] 
                              + "%.2f," % direction[1]
                              + "%.2f," % direction[2]
                              + ").vtk")
        self.right_traj_mesh.points = (rot_mat.T @ self.right_traj_mesh.points.T).T
        self.right_traj_mesh.save(self.human +  self.save_folder + "/right_traj" + "%.2f" % self.cut_translation[2] 
                              + "(%.2f," % direction[0] 
                              + "%.2f," % direction[1]
                              + "%.2f," % direction[2]
                              + ").vtk")
        
        self.point_cloud = self.point_cloud.clip(-self.cut_direction, -(self.slice_thickness - 2) * self.cut_direction + cut_pos)
        self.point_cloud.points = (rot_mat.T @ self.point_cloud.points.T).T
        self.point_cloud.save(self.human +  self.save_folder + "/point_cloud.vtk")

        if visualize:
            p = pv.Plotter()
            p.add_mesh(self.no_go_slice, color = 'r', opacity=0.2)
            p.add_mesh(self.restricted_slice, color = 'pink',opacity=0.1)
            p.add_mesh(self.cortical_slice, color = 'black', opacity=0.5)
            p.add_mesh(self.cancellous_slice, color = 'w', opacity=0.8)
            p.add_mesh(self.can_be_drilled_slice, color = 'orange', opacity=0.1)
            # trajectories
            # p.add_mesh(self.left_traj_mesh, color = 'green', opacity = 0.9)
            # p.add_mesh(self.right_traj_mesh, color = 'green', opacity=0.9)
            p.add_mesh(self.point_cloud, color = 'y', opacity=0.5)
            p.set_background([1.0,1.0,1.0])
            p.show()
    '''
    remove mesh and voxels to save space
    '''
    def clean_voxel_mesh(self):
        del(self.no_go_voxel)
        del(self.no_go_region)
        del(self.restricted_voxel)
        del(self.restricted_drilled)
        del(self.can_be_drilled)
        del(self.can_be_drilled_voxel)
        del(self.cancellous_voxel)
        del(self.cortical_voxel)
        del(self.cancellous_drilled)
        gc.collect()


    def read_reconstruction_pcd(self):
        filename = self.human + "reconstruction.pcd"
        pcd = o3d.io.read_point_cloud(filename)
        print(pcd)

        points = np.asarray(pcd.points)

        points = np.stack((points[:,2], points[:,1], points[:,0])).T - np.asarray((200, 200, 200))

        points_origin = points + self.label_map_to_origin_trans

        self.point_cloud = pv.PolyData(points_origin)

        p = pv.Plotter()
        origin_ball = pv.Sphere(20)
        p.add_mesh(self.no_go_voxel, color = 'r', opacity=0.3)
        p.add_mesh(self.restricted_voxel, color = 'pink',opacity=0.3)
        p.add_mesh(self.cortical_voxel, color = 'black', opacity=0.5)
        p.add_mesh(self.cancellous_voxel, color = 'w', lighting=1.0)
        p.add_mesh(self.can_be_drilled_voxel, color = 'orange', opacity=0.1)
        p.add_mesh(self.point_cloud, color='green', opacity=1.0)
        p.add_mesh(origin_ball)
        # p.add_mesh(bone_label_map, opacity=0.01)
        p.show_axes()
        p.show()

        


    def read_reconstruction_h5(self):
        filename = self.human + "reconstruction.h5"

        with h5py.File(filename, "r") as f:
            # Print all root level object names (aka keys) 
            # these can be group or dataset names 
            print("Keys: %s" % f.keys())
            # get first object name/key; may or may NOT be a group
            a_group_key = list(f.keys())[0]

            self.all_rec_label = f.get(a_group_key)[0, 0, :]
        
            points_list = np.where(self.all_rec_label==1)

            points = np.stack((points_list[0], points_list[1], points_list[2])).T.astype(np.float32)


            size = np.asarray(self.all_rec_label.shape, dtype=np.float32).reshape((-1, 3))
            points -=  size / 2
            print(points)

            points *= np.asarray(f.get(list(f.keys())[2]))

        
            # label map to centered model
            points_centered = points
            # centered model to origin model
            points_origin = points_centered + self.label_map_to_origin_trans
            #
        
        # move back to original place 
        self.point_cloud = pv.PolyData(points_origin)

        # # read label map
        # duke_dcm = dicom.read_file(self.human + "label_map.dcm").pixel_array

        # label_map = np.where(duke_dcm==13)
        # bone_points = np.stack((label_map)).T.astype(np.float32)
        # bone_points -= np.asarray(duke_dcm.shape) / 2
        # # bone_points += self.label_map_to_origin_trans

        # bone_label_map = pv.PolyData(bone_points)


        
        p = pv.Plotter()
        origin_ball = pv.Sphere(20)
        p.add_mesh(self.no_go_voxel, color = 'r', opacity=0.3)
        p.add_mesh(self.restricted_voxel, color = 'pink',opacity=0.3)
        p.add_mesh(self.cortical_voxel, color = 'black', opacity=0.5)
        p.add_mesh(self.cancellous_voxel, color = 'w', lighting=1.0)
        p.add_mesh(self.can_be_drilled_voxel, color = 'orange', opacity=0.1)
        p.add_mesh(self.point_cloud, color='green', opacity=0.01)
        p.add_mesh(origin_ball)
        # p.add_mesh(bone_label_map, opacity=0.01)
        p.show_axes()
        p.show()

