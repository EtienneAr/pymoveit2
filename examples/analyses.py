import os
import pickle
import numpy as np
from copy import deepcopy
import pinocchio as pin

def transform_to_se3(position, rotation_matrix):
    return pin.XYZQUATToSE3([trans.translation.x, trans.translation.y, trans.translation.z,
                             trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w])

def compute_barycenter(pose_list, fp_iter=100, callback=None):
    """
    bi invariant barycenter
    """
    guess = pose_list[0]
    if callback is not None:
        callback(guess)

    for _ in range(fp_iter):
        guess =  pin.exp(
            pin.Motion(
                np.stack([pin.log(p * guess.inverse()).np for p in pose_list], axis=0).mean(axis=0)
            )
        ) * guess
        if callback is not None:
            callback(guess)
    return guess

def compute_covariance(barycenter, pose_list):
    N = len(pose_list)
    logs_l_riem = np.stack([
        np.concatenate([
            pin.log3((barycenter.inverse() * p).rotation),
            (barycenter.inverse() * p).translation
        ], axis=0)
        for p in pose_list
    ], axis=0)
    V_l_riem = np.einsum('ib,ic->bc', logs_l_riem, logs_l_riem) / (N-1)
    return V_l_riem

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_transformations(directory):
    shoulder_M_base = None
    for i in range(4):
        file_path = os.path.join(directory, f"{i}.pkl")
        data = load_pickle(file_path)

        # Extract transformations
        world_M_tool_mocap = pin.SE3(np.reshape(np.array(data['mocap_tool_rotmats'][0]), [3,3]), np.array(data['mocap_tool_positions'][0]))
        world_M_shoulder = pin.SE3(np.reshape(np.array(data['mocap_shoulder_rotmats'][0]), [3,3]), np.array(data['mocap_shoulder_positions'][0]))
        base_M_tool_tf = pin.XYZQUATToSE3(np.array(data['tf_positions'][0] + data['tf_orientations'][0]))
        base_M_tool_tf.translation *= 1000.

        # Compute mocap (tool -> shoulder)
        shoulder_M_tool_mocap = world_M_shoulder.inverse() * world_M_tool_mocap
        tool_mocap_M_shoulder = world_M_tool_mocap.inverse() * world_M_shoulder

        if i == 0:
            shoulder_M_base = deepcopy(shoulder_M_tool_mocap * base_M_tool_tf.inverse())
            # print(f"Standard TF Transformation (Mocap -> TF) from file {i}.pkl:\n", shoulder_M_base)
        # Compute deviation from standard
        tool_tf_M_tool_mocap = tool_mocap_M_shoulder * shoulder_M_base * base_M_tool_tf
        print(f"File {i}.pkl (Deviation from standard TF):\n", tool_tf_M_tool_mocap)

if __name__ == "__main__":
    directory = "/home/earlaud/exchange/measures/2025-04-02_14-41-55"  # Change to your actual directory
    compute_transformations(directory)