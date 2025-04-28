import os
import pickle
import numpy as np
from copy import deepcopy
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize

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

def visualize_transformations(*transformations):
    """Visualizes transformations in 3D space."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for transform_list in transformations:
        ax.scatter([t_a.translation[0] for t_a in transform_list],
                [t_a.translation[1] for t_a in transform_list],
                [t_a.translation[2] for t_a in transform_list])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def extract_all_transforms():
    # Extract transformations for each file and each time step
    mocap_world_M_tool_list = []
    mocap_world_M_shoulder_list = []
    fk_base_M_tool_list = []
    for file_nb in range(38):
        file_path = os.path.join("/home/earlaud/exchange/measures/2025-04-02_17-13-45", f"{file_nb}.pkl")
        data = load_pickle(file_path)
        for j in range(len(data['mocap_tool_rotmats'])):
            world_M_tool = pin.SE3(np.reshape(np.array(data['mocap_tool_rotmats'][j]), [3,3]), np.array(data['mocap_tool_positions'][j]))
            world_M_tool.translation[2] = 0.
            tool_correction = pin.XYZQUATToSE3(np.array([0,0,0] + [ 0.5, -0.5, 0.5, 0.5 ]))
            world_M_tool = world_M_tool * tool_correction

            world_M_shoulder = pin.SE3(np.reshape(np.array(data['mocap_shoulder_rotmats'][j]), [3,3]), np.array(data['mocap_shoulder_positions'][j]))
            world_M_shoulder.translation[2] = 0.

            base_M_tool_tf = pin.XYZQUATToSE3(np.array(data['tf_positions'][j] + data['tf_orientations'][j]))
            base_M_tool_tf.translation *= 1000.

            mocap_world_M_tool_list.append(world_M_tool)
            mocap_world_M_shoulder_list.append(world_M_shoulder)
            fk_base_M_tool_list.append(base_M_tool_tf)

    return mocap_world_M_shoulder_list, mocap_world_M_tool_list, fk_base_M_tool_list

def compute_shoulder_M_base_bruteforce(world_M_shoulder_list, world_M_tool_list, base_M_tool_list, initial_guess):
    def guess_distance(xyz_rpy_alpha, display = False):
        shoulder_M_base_guess = pin.exp(np.array(xyz_rpy_alpha[:6]))

        # Compute all the individual errors.
        errs = []
        world_M_tool_pred_list = []
        for world_M_shoulder, world_M_tool, base_M_tool in zip(world_M_shoulder_list, world_M_tool_list, base_M_tool_list):
            world_M_shoulder.translation[2] = xyz_rpy_alpha[6]
            world_M_tool_pred = world_M_shoulder * shoulder_M_base_guess * base_M_tool

            if(display):
                world_M_tool_pred.translation[2] = 0.
                world_M_tool_pred_list.append(world_M_tool_pred)

            err_trans = np.array([ world_M_tool.translation[0] - world_M_tool_pred.translation[0]
                                 , world_M_tool.translation[1] - world_M_tool_pred.translation[1]])

            err_rot = pin.log3((world_M_tool * world_M_tool_pred.inverse()).rotation)

            err = np.linalg.norm(np.concatenate((err_trans, err_rot)))

            errs.append(err)

        if(display):
            visualize_transformations(world_M_tool_list, world_M_tool_pred_list)

        return np.linalg.norm(errs)

    optim_res = minimize(guess_distance, initial_guess, method="BFGS")
    print(optim_res)
    guess_distance(optim_res.x, display=True)

def compute_world_M_base_inexact(world_M_shoulder_list, world_M_tool_list, base_M_tool_list):
    world_M_base_list = []
    for world_M_tool, base_M_tool in zip(world_M_tool_list, base_M_tool_list):
        world_M_base = world_M_tool * base_M_tool.inverse()
        world_M_base_list.append(world_M_base)

    w_M_b = compute_barycenter(world_M_base_list)

    # visualize_transformations(world_M_tool_list, [ w_M_b * base_M_tool for base_M_tool in base_M_tool_list])

    return w_M_b

def analyse_4x4x4_experiment():
    np.set_printoptions(formatter={'float': lambda x: f"{x:.0e}"})

    mocap_world_M_shoulder_list, mocap_world_M_tool_list, fk_base_M_tool_list = extract_all_transforms()

    w_M_b = compute_world_M_base_inexact(mocap_world_M_shoulder_list, mocap_world_M_tool_list, fk_base_M_tool_list)

    s_M_b = mocap_world_M_shoulder_list[0].inverse() * w_M_b
    initial_guess = pin.log(s_M_b).np.tolist() + [0.]

    compute_shoulder_M_base_bruteforce(mocap_world_M_shoulder_list, mocap_world_M_tool_list, fk_base_M_tool_list, initial_guess)

if __name__ == "__main__":
    analyse_4x4x4_experiment()