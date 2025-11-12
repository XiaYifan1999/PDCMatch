import igl
from datasets.shape_dataset import *
from utils.geometry_util import torch2np

def harmonic_interpolation(V, F, boundary_indices, boundary_values):
    L = igl.cotmatrix(V, F)
    n = V.shape[0]
    interior_indices = np.setdiff1d(np.arange(n), boundary_indices)
    A = L[interior_indices][:, interior_indices]
    b = -L[interior_indices][:, boundary_indices] @ boundary_values
    u_interior = scipy.sparse.linalg.spsolve(A, b)
    u = np.zeros((n, boundary_values.shape[1]))
    u[boundary_indices] = boundary_values
    u[interior_indices] = u_interior
    return u

def get_orientation_calibration_matrix(up_vector, front_vector):
    # align right, up, front dir of the input shape with/into x, y, z axis
    right_vector = np.cross(up_vector, front_vector)
    assert not np.allclose(right_vector, 0) # ensure no degenerate input
    matrix = np.column_stack((right_vector, up_vector, front_vector))
    return matrix

def rotation_matrix_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c,  0, s],
        [ 0,  1, 0],
        [-s,  0, c]
    ])
import numpy as np

def rotation_matrix_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])
    
def rotation_matrix_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])
    
def orientation_calibration_by_dataset(test_set):
    if type(test_set) == PairFaustDataset: # y up, z front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0,1,0], front_vector=[0,0,1]) 
    elif type(test_set) == PairScapeDataset:
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0,1,0], front_vector=[1,0,0]) 
    elif type(test_set) == PairShrec19Dataset:
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0,1,0], front_vector=[0,0,1])
        angle = np.radians(-180)  # 例如15 表示逆时针15度
        rotation_adjust = rotation_matrix_y(angle)
        orientation_matrix = rotation_adjust @ orientation_matrix
    elif type(test_set) == PairSmalDataset: # neg y up, z front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0,-1,0], front_vector=[0,0,1]) 
        # # 添加一个绕 Z 轴的微调旋转（逆时针，角度以弧度表示）
        angle = np.radians(60)  # 例如15 表示逆时针15度
        rotation_adjust = rotation_matrix_y(angle)
        orientation_matrix = rotation_adjust @ orientation_matrix
    elif type(test_set) == PairDT4DDataset: # z up, neg y front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0,0,1], front_vector=[0,-1,0]) 
        angle_deg = -30  # 例如逆时针转10°
        rotation_adjust = rotation_matrix_x(np.radians(angle_deg))
        orientation_matrix = rotation_adjust @ orientation_matrix
    elif type(test_set) == PairTopKidsDataset: # z up, neg y front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0,0,1], front_vector=[0,-1,0]) 
    else:
        print("Unimplemented dataset type, use default orientation matrix y up, z front")
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0,1,0], front_vector=[0,0,1])
    return orientation_matrix


def limbs_indices_by_dataset(i, data_x, test_set):
    if type(test_set) == PairFaustDataset:
        landmarks = np.array([4962, 1249, 77, 2523,2185])
        limbs_indices = torch2np(data_x["corr"])[landmarks]
    elif type(test_set) == PairSmalDataset:
        landmarks = np.array([1198, 2324, 2523, 1939, 30, 28])  # [hands, feet, head, tail]
        limbs_indices = torch2np(data_x["corr"])[landmarks]
    elif type(test_set) == PairDT4DDataset:
        landmarks = np.array([4819, 2932, 3003, 4091, 3843])  # [hands, feet, head, tail]
        first_index, _ = test_set.combinations[i]
        first_cat = test_set.dataset.off_files[first_index].split('/')[-2]
        if first_cat == 'crypto':
            corr = torch2np(data_x["corr"])
        elif first_cat == 'mousey':
            landmarks = np.array([2583, 6035, 3028, 6880, 78])
            corr = torch2np(data_x["corr"])
        elif first_cat == 'ortiz':
            landmarks = np.array([3451, 7721, 161, 1424, 2021])
            corr = torch2np(data_x["corr"])
        else:
            corr_inter = np.loadtxt(os.path.join(test_set.dataset.data_root, 'corres', 'cross_category_corres',
                                        f'crypto_{first_cat}.vts'), dtype=np.int32) - 1
            corr_intra = torch2np(data_x["corr"])
            corr = corr_intra[corr_inter]
        limbs_indices = corr[landmarks]
    elif type(test_set) == PairTopKidsDataset:
        limbs_indices = np.array([8438, 7998, 11090, 11416, 9885])  # [hands, feet, head
    else:
        print("Unimplemented dataset type, colored limb indices(hands, feets, head, tail) are not defined.")
        limbs_indices = np.arange(6)
    return limbs_indices