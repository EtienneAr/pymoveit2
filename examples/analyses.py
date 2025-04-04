import os
import pickle
import numpy as np
from copy import deepcopy
import pinocchio as pin
import matplotlib.pyplot as plt

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

def compute_transforms_avg_from_file(file_path, shoulder_M_base = None):
    data = load_pickle(file_path)

    # Extract transformations for each time step
    result_list = []
    for i in range(len(data['mocap_tool_rotmats'])):
        world_M_tool_mocap = pin.SE3(np.reshape(np.array(data['mocap_tool_rotmats'][0]), [3,3]), np.array(data['mocap_tool_positions'][0]))
        world_M_shoulder = pin.SE3(np.reshape(np.array(data['mocap_shoulder_rotmats'][0]), [3,3]), np.array(data['mocap_shoulder_positions'][0]))
        base_M_tool_tf = pin.XYZQUATToSE3(np.array(data['tf_positions'][0] + data['tf_orientations'][0]))
        base_M_tool_tf.translation *= 1000.

        # Compute mocap (tool -> shoulder)
        shoulder_M_tool_mocap = world_M_shoulder.inverse() * world_M_tool_mocap
        tool_mocap_M_shoulder = world_M_tool_mocap.inverse() * world_M_shoulder

        # If no shoulder to base tranform given, compute it !
        # -> Hyp: tf and mocap are located at the same place
        if not shoulder_M_base:
            shoulder_M_base_i = shoulder_M_tool_mocap * base_M_tool_tf.inverse()
            result_list.append(shoulder_M_base_i)
        else:
            # Compute mocap vs tf error
            tool_tf_M_tool_mocap = tool_mocap_M_shoulder * shoulder_M_base * base_M_tool_tf
            result_list.append(tool_tf_M_tool_mocap)

    result_avg = compute_barycenter(result_list)
    result_variance = compute_covariance(result_avg, result_list)
    return result_avg, result_variance


def analyse_4x4x4_0123_experiment():
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    x = [0., 1., 2., 3.]
    for i in range(24):
        file_path = os.path.join("/home/earlaud/exchange/measures/2025-04-02_17-36-56", f"{i}.pkl")

        if (i %4) == 0:
            # Reset measurements for plots
            y = []
            y_err = []
            # Compute shoulder to base transform (for reference)
            shoulder_M_base, shoulder_M_base_variance = compute_transforms_avg_from_file(file_path, None)

        # Compute Mocap vs tf transform
        tool_tf_M_tool_mocap, tool_tf_M_tool_mocap_variance = compute_transforms_avg_from_file(file_path, shoulder_M_base)
        ecart_type = np.sqrt(np.linalg.norm(tool_tf_M_tool_mocap_variance[3:,3:]))

        y.append(np.linalg.norm(tool_tf_M_tool_mocap.translation))
        y_err.append(ecart_type)

        if (i %4) == 3:
            ax.errorbar(x, y, yerr=y_err, fmt='-o', label=f'configuration {i//4 +1}', capsize=3)
            plt.draw()

    plt.ioff()  # Turn off interactive mode
    plt.ylabel("Deviation (mm)")
    plt.xlabel("Weight (kg)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    analyse_4x4x4_0123_experiment()