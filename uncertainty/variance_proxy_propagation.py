import numpy as np
import skgeom as sg
from skgeom import minkowski, PolygonWithHoles
import torch
from torch import nn
import scipy
from .estimate_variance_proxy import*



def vp_propagation(vp_0, vp_w, vp_e, A, B, C, K, Ls, T):
    '''
    propagate variance proxy over the trajectory
    - vp_0: init x variance proxy
    - vp_w: disturbance noise variance proxy
    - vp_e: measurement noise variance proxy
    - A, B, C: system matrices
    - K: control matrices
    - Ls: observer matrices
    - T: time horizon
    '''
    
    L = Ls[0]

    # define matrixes
    I = np.eye(A.shape[0])
    O_est = np.zeros((A.shape[0], L.shape[1]))
    O_track = np.zeros((A.shape[0], A.shape[0]))

    # init matrices
    M = (I - L @ C)
    A_est = M @ A

    A_track_up = np.concatenate([A, B @ K], axis=1)
    A_track_down = np.concatenate([L @ C @ A, A + B @ K - L @ C @ A], axis=1)
    A_track = np.concatenate([A_track_up, A_track_down], axis=0)
    B_track = np.concatenate([I, L @ C], axis=0)
    C_track = np.concatenate([O_est, L], axis=0)
    

    # initialization
    vp_est = M @ vp_0 @ M.T + L @ vp_e @ L.T
    vp_est_track = L @ C @ vp_0 @ C.T @ L.T + L @ vp_e @ L.T
    vp_track_up = np.concatenate([vp_0, M @ vp_0 @ C.T @ L.T], axis=1)
    vp_track_down = np.concatenate([L @ C @ vp_0 @ M.T, vp_est_track], axis=1)
    vp_track = np.concatenate([vp_track_up, vp_track_down], axis=0)

    vp_ests = [vp_est]
    vp_tracks = [vp_track]

    # propagate variance proxy
    for t in range(T):
        L = Ls[t+1]

        # update matrices
        M = (I - L @ C)
        A_est = M @ A

        A_track_up = np.concatenate([A, B @ K], axis=1)
        A_track_down = np.concatenate([L @ C @ A, A + B @ K - L @ C @ A], axis=1)
        A_track = np.concatenate([A_track_up, A_track_down], axis=0)
        B_track = np.concatenate([I, L @ C], axis=0)
        C_track = np.concatenate([O_est, L], axis=0)

        # propagate
        vp_est = A_est @ vp_est @ A_est.T + L @ vp_e @ L.T + M @ vp_w @ M.T
        vp_track = A_track @ vp_track @ A_track.T + B_track @ vp_w @ B_track.T + C_track @ vp_e @ C_track.T

        vp_ests.append(vp_est)
        vp_tracks.append(vp_track)

    return vp_ests, vp_tracks


def vp_propagation_est_track(vp_0, vp_w, vp_e, A, B, C, K, Ls, T):
    '''
    propagate variance proxy over the trajectory
    - vp_0: init x variance proxy
    - vp_w: disturbance noise variance proxy
    - vp_e: measurement noise variance proxy
    - A, B, C: system matrices
    - K: control matrices
    - Ls: observer matrices
    - T: time horizon
    '''
    
    L = Ls[0]

    # define matrixes
    I = np.eye(A.shape[0])
    O_est = np.zeros((A.shape[0], L.shape[1]))
    O_track = np.zeros((A.shape[0], A.shape[0]))

    # init matrices
    M = (I - L @ C)
    A_est = M @ A


    # initialization
    vp_est = M @ vp_0 @ M.T + L @ vp_e @ L.T
    vp_track_up = np.concatenate([vp_est, M @ vp_0], axis=1)
    # e_est; e_track_nominal_true
    vp_track_down = np.concatenate([vp_0.T @ M.T, vp_0], axis=1)
    vp_track = np.concatenate([vp_track_up, vp_track_down], axis=0)

    vp_ests = [vp_est]
    vp_tracks = [vp_track]

    # propagate variance proxy
    for t in range(T):
        L = Ls[t+1]

        # update matrices
        M = (I - L @ C)
        A_est = M @ A

        A_track_up = np.concatenate([A_est, O_track], axis=1)
        A_track_down = np.concatenate([-B @ K, A + B @ K], axis=1)
        A_track = np.concatenate([A_track_up, A_track_down], axis=0)
        B_track = np.concatenate([M, I], axis=0)
        C_track = np.concatenate([L, O_est], axis=0)

        # propagate
        vp_est = A_est @ vp_est @ A_est.T + L @ vp_e @ L.T + M @ vp_w @ M.T
        vp_track = A_track @ vp_track @ A_track.T + B_track @ vp_w @ B_track.T + C_track @ vp_e @ C_track.T

        vp_ests.append(vp_est)
        vp_tracks.append(vp_track)

    # vp_track:
    # up: est error
    # down: true to nominal

    return vp_ests, vp_tracks


def linear_transform_poly(A, poly):
    poly = sg.Polygon((A @ poly.coords.T).T)
    # if poly.orientation():
    #     poly = sg.Polygon((-A @ poly.coords.T).T)
    # while poly.is_simple() == False:  
    #     poly = sg.simplify(poly, 0.7)#
    return poly


def robust_propagation_2d(
        poly_0: sg.Polygon, 
        poly_w: sg.Polygon, 
        poly_e: sg.Polygon, 
        A, B, C, K, Ls, T):
    '''
    propagate robust set over the trajectory
    - poly_0: init x polygon
    - poly_w: disturbance noise polygon
    - poly_e: measurement noise polygon
    - A, B, C: system matrices
    - K: control matrices
    - Ls: observer matrices
    - T: time horizon
    '''
    # if A.shape != (2, 2):
    #     raise ValueError("A should be 2x2 matrix")

    # define matrixes
    I = np.eye(A.shape[0])
    O = np.zeros(A.shape)

    L = Ls[0]

    # init the error set
    poly_est = minkowski.minkowski_sum(
        linear_transform_poly(I - L @ C, poly_0), 
        linear_transform_poly(L, poly_e)).outer_boundary()
    
    poly_track_est = minkowski.minkowski_sum(
        linear_transform_poly(L @ C, poly_0),
        linear_transform_poly(L, poly_e)).outer_boundary() # est to nominal
    poly_track_true = poly_0

    poly_track_ests = [poly_track_est]
    poly_track_trues = [poly_track_true]
    poly_ests = [poly_est]
    # propagate the error set
    print('robust set propagation')
    for t in range(T):
        print(t)
        L = Ls[t+1]

        # update matrices
        # while len(list(poly_est.vertices)) > 20 or not poly_est.is_simple():
        #     poly_est = sg.simplify(poly_est, 0.5)

        # propagate
        try:
            poly_est = minkowski.minkowski_sum(
                linear_transform_poly((I - L @ C) @ A, poly_est), 
                linear_transform_poly(L, poly_e)).outer_boundary()
        
            poly_est = minkowski.minkowski_sum(
                poly_est, 
                linear_transform_poly((I - L @ C), poly_w)).outer_boundary()
        except:
            poly_est = poly_est


        # while len(list(poly_est.vertices)) > 20 or not poly_est.is_simple():
        #     poly_est = sg.simplify(poly_est, 0.5)
        
        
        try:
            poly_track_true = minkowski.minkowski_sum(
                linear_transform_poly(A + B @ K, poly_track_true),
                linear_transform_poly(B @ K, poly_est)).outer_boundary()
    
                
            poly_track_true = minkowski.minkowski_sum(
                poly_track_true,
                poly_w).outer_boundary()
        except:
            poly_track_true = poly_track_true
        
 
        
        
        try:
            poly_track_est = minkowski.minkowski_sum(
                linear_transform_poly(L @ C @ A, poly_est),
                linear_transform_poly((A + B @ K), poly_track_est)).outer_boundary()
            
            poly_track_est = minkowski.minkowski_sum(         
                poly_track_est,
                linear_transform_poly(L @ C, poly_w)).outer_boundary()
            
            poly_track_est = minkowski.minkowski_sum(
                poly_track_est,
                linear_transform_poly(L, poly_e)).outer_boundary()
        except:
            poly_track_est = poly_track_est
        

        
        # for not simply poligon
        while len(list(poly_est.vertices)) > 30 or not poly_est.is_simple():
            poly_est = sg.simplify(poly_est, 0.8)
        while len(list(poly_track_est.vertices)) > 30 or not poly_track_est.is_simple():
            poly_track_est = sg.simplify(poly_track_est, 0.8)
        while len(list(poly_track_true.vertices)) > 30 or not poly_track_true.is_simple():
            poly_track_true = sg.simplify(poly_track_true, 0.8)

        
        # if not poly_track_est.is_simple():
        
        poly_ests.append(poly_est)
        poly_track_trues.append(poly_track_true)
        poly_track_ests.append(poly_track_est)

    return poly_ests, poly_track_ests, poly_track_trues



def points_minkowski_sum(poly_1, poly_2):

    # broad cast
    broad_points = poly_1.reshape(-1, 1, poly_1.shape[-1]) + poly_2.reshape(1, -1, poly_2.shape[-1])
    broad_points = broad_points.reshape(-1, poly_1.shape[-1])

    # get convex hull
    hull = scipy.spatial.ConvexHull(broad_points)
    # if len(hull.vertices) > 100:
    #     return find_max_min_points(broad_points[hull.vertices, :])

    # remove too close points
    
        
    return broad_points[hull.vertices, :]
    # 


def find_max_min_points(points):
    """
    Find the points with the maximum and minimum values along each dimension.

    Parameters:
    points (numpy.ndarray): An (N, n) array where N is the number of points and n is the dimensionality.

    Returns:
    dict: A dictionary with keys 'max' and 'min', each containing a (n, n) array.
          The 'max' array contains points with the maximum values in each dimension.
          The 'min' array contains points with the minimum values in each dimension.
    """
    # Find the indices of the max and min values along each dimension
    max_indices = np.argmax(points, axis=0)
    min_indices = np.argmin(points, axis=0)
    
    # Extract the points corresponding to these indices
    max_points = points[max_indices, :]
    min_points = points[min_indices, :]
    
    return np.concatenate([max_points, min_points], axis=0)


def robust_propagation(
        poly_0: np.ndarray, 
        poly_w: np.ndarray, 
        poly_e: np.ndarray, 
        A, B, C, K, Ls, T):
    '''
    propagate robust set over the trajectory
    - poly_0: init x polygon (N_1, d)
    - poly_w: disturbance noise polygon (N_2, d)
    - poly_e: measurement noise polygon (N_3, d)
    - A, B, C: system matrices
    - K: control matrices
    - Ls: observer matrices
    - T: time horizon
    '''
    # if A.shape != (2, 2):
    #     raise ValueError("A should be 2x2 matrix")

    # define matrixes
    I = np.eye(A.shape[0])
    O = np.zeros(A.shape)

    L = Ls[0]

    # init the error set
    poly_est = points_minkowski_sum(
        poly_0 @ (I - L @ C).T, 
        poly_e @ L.T
    ) # est to nominal

    
    poly_track_est = points_minkowski_sum(
        poly_0 @ (L @ C).T,
        poly_e @ L.T # est to nominal
    )

    poly_track_true = poly_0

    poly_track_ests = [poly_track_est]
    poly_track_trues = [poly_track_true]
    poly_ests = [poly_est]
    # propagate the error set
    print('robust set propagation')
    for t in range(T):
        print(t)
        L = Ls[t+1]

        poly_est = points_minkowski_sum(
            ((I - L @ C) @ A @ poly_est.T).T, 
            (L @ poly_e.T).T
        )
    
        poly_est = points_minkowski_sum(
            poly_est, 
            ((I - L @ C) @ poly_w.T).T
        )
        
        
        poly_track_true = points_minkowski_sum(
            ((A + B @ K) @ poly_track_true.T).T,
            (B @ K @ poly_est.T).T
        )

            
        poly_track_true = points_minkowski_sum(
            poly_track_true,
            poly_w
        )
    
 
        poly_track_est = points_minkowski_sum(
            (L @ C @ A @ poly_est.T).T,
            ((A + B @ K) @ poly_track_est.T).T
        )
        
        poly_track_est = points_minkowski_sum(         
            poly_track_est,
            (L @ C @ poly_w.T).T
        )
        
        poly_track_est = points_minkowski_sum(
            poly_track_est,
            (L @ poly_e.T).T
        )
        
        
        poly_ests.append(poly_est)
        poly_track_trues.append(poly_track_true)
        poly_track_ests.append(poly_track_est)

    return poly_ests, poly_track_ests, poly_track_trues


def extract_vp_track_down(vp_track, dim):
    select = np.concatenate([ np.zeros((dim, dim)), np.eye(dim)], axis=1)
    vp_track_est_to_nominal = select @ vp_track @ select.T
    return vp_track_est_to_nominal

def extract_vp_track_up(vp_track, dim):
    select = np.concatenate([np.eye(dim), np.zeros((dim, dim))], axis=1)
    vp_track_true_to_nominal = select @ vp_track @ select.T
    return vp_track_true_to_nominal

    

def optimal_propagation(M_list):
    a_list = [np.sqrt(np.trace(M)) for M in M_list]
    
    result = 0
    for i in range(len(M_list)):
        result += 1 / a_list[i] * M_list[i]

    result *= sum(a_list)

    return result


def sub_gau_norm_propagation(sg_0, sg_w, sg_e, A, B, C, K, Ls, T):
    '''
    propagate variance proxy over the trajectory
    - sg_0: init x sub gaussian norm
    - sg_w: disturbance noise sub gaussian norm
    - sg_e: measurement noise sub gaussian norm
    - A, B, C: system matrices
    - K: control matrices
    - Ls: observer matrices
    - T: time horizon
    '''
    # define matrixes
    I = np.eye(A.shape[0])
    O_est = np.zeros((A.shape[0], C.shape[0]))
    O_track = np.zeros((A.shape[0], A.shape[0]))

    L = Ls[0]

    # init matrices
    M = (I - L @ C)
    A_est = M @ A

    A_track_up = np.concatenate([A, B @ K], axis=1)
    A_track_down = np.concatenate([L @ C @ A, A + B @ K - L @ C @ A], axis=1)
    A_track = np.concatenate([A_track_up, A_track_down], axis=0)
    B_track = np.concatenate([I, L @ C], axis=0)
    C_track = np.concatenate([O_est, L], axis=0)
    

    # initialization
    sg_est = optimal_propagation([M @ sg_0 @ M.T, L @ sg_e @ L.T])
    sg_est_track = optimal_propagation([L @ C @ sg_0 @ C.T @ L.T, L @ sg_e @ L.T])
    sg_track_up = np.concatenate([sg_0, M @ sg_0 @ C.T @ L.T], axis=1)
    sg_track_down = np.concatenate([L @ C @ sg_0 @ M.T, sg_est_track], axis=1)
    sg_track = np.concatenate([sg_track_up, sg_track_down], axis=0)

    sg_ests = [sg_est]
    sg_tracks = [sg_track]

    # propagate variance proxy
    for t in range(T):
        L = Ls[t+1]

        # update matrices
        M = (I - L @ C)
        A_est = M @ A

        A_track_up = np.concatenate([A, B @ K], axis=1)
        A_track_down = np.concatenate([L @ C @ A, A + B @ K - L @ C @ A], axis=1)
        A_track = np.concatenate([A_track_up, A_track_down], axis=0)
        B_track = np.concatenate([I, L @ C], axis=0)
        C_track = np.concatenate([O_est, L], axis=0)

        # propagate
        sg_est = optimal_propagation([A_est @ sg_est @ A_est.T, L @ sg_e @ L.T, M @ sg_w @ M.T])
        sg_track = optimal_propagation([A_track @ sg_track @ A_track.T, B_track @ sg_w @ B_track.T, C_track @ sg_e @ C_track.T])

        sg_ests.append(sg_est)
        sg_tracks.append(sg_track)

    # track error:
    # up: true to nominal
    # down: est to nominal

    return sg_ests, sg_tracks



def robust_ellipsoid_propagation(sg_0, sg_w, sg_e, A, B, C, K, Ls, T):
    '''
    propagate variance proxy over the trajectory
    - sg_0: init x sub gaussian norm
    - sg_w: disturbance noise sub gaussian norm
    - sg_e: measurement noise sub gaussian norm
    - A, B, C: system matrices
    - K: control matrices
    - Ls: observer matrices
    - T: time horizon
    '''
    # define matrixes
    L = Ls[0]

    # define matrixes
    I = np.eye(A.shape[0])
    O_est = np.zeros((A.shape[0], L.shape[1]))
    O_track = np.zeros((A.shape[0], A.shape[0]))

    # init matrices
    M = (I - L @ C)
    A_est = M @ A


    # initialization
    sg_est = optimal_propagation([M @ sg_0 @ M.T, L @ sg_e @ L.T])
    sg_track_up = np.concatenate([sg_est, M @ sg_0], axis=1)
    # e_est; e_track_nominal_true
    sg_track_down = np.concatenate([sg_0.T @ M.T, sg_0], axis=1)
    sg_track = np.concatenate([sg_track_up, sg_track_down], axis=0)

    sg_ests = [sg_est]
    sg_tracks = [sg_track]

    # propagate variance proxy
    for t in range(T):
        L = Ls[t+1]

        # update matrices
        M = (I - L @ C)
        A_est = M @ A

        A_track_up = np.concatenate([A_est, O_track], axis=1)
        A_track_down = np.concatenate([-B @ K, A + B @ K], axis=1)
        A_track = np.concatenate([A_track_up, A_track_down], axis=0)
        B_track = np.concatenate([M, I], axis=0)
        C_track = np.concatenate([L, O_est], axis=0)

        # propagate
        sg_est = optimal_propagation([A_est @ sg_est @ A_est.T, L @ sg_e @ L.T, M @ sg_w @ M.T])
        sg_track = optimal_propagation([A_track @ sg_track @ A_track.T, B_track @ sg_w @ B_track.T, C_track @ sg_e @ C_track.T])

        sg_ests.append(sg_est)
        sg_tracks.append(sg_track)

        # sg_track:
        # up: est error
        # down: true to nominal


    return sg_ests, sg_tracks

