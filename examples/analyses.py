import os
import pickle
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_transformation_matrix(position, orientation):
    """Creates a 4x4 transformation matrix from position and quaternion orientation."""
    rotmatrix = np.reshape(np.array(orientation), [3,3])
    transform = np.eye(4)
    transform[:3, :3] = rotmatrix
    transform[:3, 3] = position
    return transform

def create_transformation_matrix_from_quat(position, quat):
    """Creates a 4x4 transformation matrix from position and quaternion orientation."""
    rot_matrix = R.from_quat(quat).as_matrix()
    return create_transformation_matrix(position, rot_matrix)

def invert_homogeneous_matrix(T):
    """Computes the inverse of a homogeneous transformation matrix analytically."""
    R_inv = T[:3, :3].T  # Transpose of the rotation matrix (inverse of a rotation matrix)
    t_inv = -R_inv @ T[:3, 3]  # Compute the inverse translation

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv

def visualize_transformations(transformations, labels):
    """Visualizes transformations in 3D space."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for T, label in zip(transformations, labels):
        position = T[:3, 3]
        ax.scatter(position[0], position[1], position[2], label=label)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def compute_transformations(directory):
    shoulder_M_base = None
    for i in range(4):
        file_path = os.path.join(directory, f"{i}.pkl")
        data = load_pickle(file_path)

        # Extract transformations
        world_M_tool_mocap = create_transformation_matrix(data['mocap_tool_positions'][0], data['mocap_tool_rotmats'][0])
        world_M_shoulder = create_transformation_matrix(data['mocap_shoulder_positions'][0], data['mocap_shoulder_rotmats'][0])
        base_M_tool_tf = create_transformation_matrix_from_quat(data['tf_positions'][0], data['tf_orientations'][0])
        base_M_tool_tf[:3, 3] *= 1000.

        # Compute mocap (tool -> shoulder)
        shoulder_M_tool_mocap = invert_homogeneous_matrix(world_M_shoulder) @ world_M_tool_mocap
        tool_mocap_M_shoulder = invert_homogeneous_matrix(world_M_tool_mocap) @ world_M_shoulder

        if i == 0:
            shoulder_M_base = shoulder_M_tool_mocap @ invert_homogeneous_matrix(base_M_tool_tf)
            # print(f"Standard TF Transformation (Mocap -> TF) from file {i}.pkl:\n", shoulder_M_base)
        # Compute deviation from standard
        tool_tf_M_tool_mocap = tool_mocap_M_shoulder @ shoulder_M_base @ base_M_tool_tf
        print(f"File {i}.pkl (Deviation from standard TF):\n", tool_tf_M_tool_mocap)

if __name__ == "__main__":
    directory = "/home/earlaud/exchange/measures/2025-04-02_14-27-58"  # Change to your actual directory
    compute_transformations(directory)