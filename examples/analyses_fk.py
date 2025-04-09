import os
import pickle
import numpy as np
from copy import deepcopy
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

def extract_all_transforms():
    # Extract transformations for each file and each time step
    mocap_shoulder_M_tool_list = []
    fk_base_M_tool_list = []
    for file_nb in range(38):
        file_path = os.path.join("/home/earlaud/exchange/measures/2025-04-02_17-13-45", f"{file_nb}.pkl")
        data = load_pickle(file_path)
        for j in range(len(data['mocap_tool_rotmats'])):
            world_M_tool_mocap = pin.SE3(np.reshape(np.array(data['mocap_tool_rotmats'][j]), [3,3]), np.array(data['mocap_tool_positions'][j]))
            world_M_shoulder = pin.SE3(np.reshape(np.array(data['mocap_shoulder_rotmats'][j]), [3,3]), np.array(data['mocap_shoulder_positions'][j]))
            base_M_tool_tf = pin.XYZQUATToSE3(np.array(data['tf_positions'][j] + data['tf_orientations'][j]))
            base_M_tool_tf.translation *= 1000.

            # Compute mocap (tool -> shoulder)
            shoulder_M_tool_mocap = world_M_shoulder.inverse() * world_M_tool_mocap

            mocap_shoulder_M_tool_list.append(shoulder_M_tool_mocap)
            fk_base_M_tool_list.append(base_M_tool_tf)

    return mocap_shoulder_M_tool_list, fk_base_M_tool_list

def compute_base_M_shoulder(shoulder_M_tool_list, base_M_tool_list):
    # Init with plausible guess
    guess_base_M_shoulder = base_M_tool_list[-1] * shoulder_M_tool_list[-1].inverse()

    for i in range(100):
        log_avg = np.zeros(6)
        log_avg_cnt = 0
        for shoulder_M_tool, base_M_tool in zip(shoulder_M_tool_list, base_M_tool_list):
            M_error = base_M_tool * shoulder_M_tool.inverse() * guess_base_M_shoulder.inverse()
            log_avg += pin.log(M_error).np
            log_avg_cnt += 1
        log_avg /= float(log_avg_cnt)
        print(log_avg)
        guess_base_M_shoulder =  pin.exp(log_avg) * guess_base_M_shoulder

    return guess_base_M_shoulder

def analyse_4x4x4_experiment():
    np.set_printoptions(formatter={'float': lambda x: f"{x:.0e}"})

    mocap_shoulder_M_tool_list, fk_base_M_tool_list = extract_all_transforms()

    # Guess the orientation of the testbench in the mocap frame
    guess_base_M_shoulder = compute_base_M_shoulder(mocap_shoulder_M_tool_list, fk_base_M_tool_list)

    # Compute avg / variance of error
    mocap_tool_M_fk_tool_list = []
    for mocap_shoulder_M_tool, fk_base_M_tool in zip(mocap_shoulder_M_tool_list, fk_base_M_tool_list):
        mocap_tool_M_fk_tool = mocap_shoulder_M_tool.inverse() * guess_base_M_shoulder.inverse() * fk_base_M_tool
        mocap_tool_M_fk_tool_list.append(mocap_tool_M_fk_tool)

    mocap_tool_M_fk_tool_avg = compute_barycenter(mocap_tool_M_fk_tool_list)
    mocap_tool_M_fk_tool_variance = compute_covariance(mocap_tool_M_fk_tool_avg, mocap_tool_M_fk_tool_list)

    print(mocap_tool_M_fk_tool_avg)
    print(np.sqrt(np.linalg.norm(mocap_tool_M_fk_tool_variance[3:,3:])))

if __name__ == "__main__":
    analyse_4x4x4_experiment()