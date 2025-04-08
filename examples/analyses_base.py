import os
import numpy as np
from copy import deepcopy
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import math


def transform_to_se3(data, obj_name):
    position = np.array([float(data[obj_name + ' X']), float(data[obj_name + ' Y']), float(data[obj_name + ' Z'])])
    rotmat = np.array([ [float(data[obj_name + ' Rot[0]']), float(data[obj_name + ' Rot[1]']), float(data[obj_name + ' Rot[2]'])],
                        [float(data[obj_name + ' Rot[3]']), float(data[obj_name + ' Rot[4]']), float(data[obj_name + ' Rot[5]'])],
                        [float(data[obj_name + ' Rot[6]']), float(data[obj_name + ' Rot[7]']), float(data[obj_name + ' Rot[8]'])]])

    return pin.SE3(rotmat, position)

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

def compute_transforms_avg_from_file(file_path):
    # Open and read the CSV
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # This automatically uses the first row as headers

        # Prepare return table
        result = []

        # Parse each line
        prev_meas_filename = None
        prev_meas_folder = None
        prev_meas_pose_id = None
        prev_meas_left_weight = None
        prev_meas_right_weight = None
        base_pose_list = []
        torso_pose_list = []
        for row in reader:
            curr_meas_filename = row['Name']
            curr_meas_folder = row['Folder']
            curr_meas_pose_id = row['PoseID']
            curr_meas_left_weight = float(row['Left_bags']) * 0.450 + float(row['Left_weight']) * 1.0
            curr_meas_right_weight = float(row['Right_bags']) * 0.450 + float(row['Right_weight']) * 1.0

            # If the end of this measurement is reached
            if curr_meas_filename != prev_meas_filename:

                if len(base_pose_list) > 0:
                    #average all the values
                    base_pose_avg = compute_barycenter(base_pose_list)
                    torso_pose_avg = compute_barycenter(torso_pose_list)
                    base_variance =compute_covariance(base_pose_avg, base_pose_list)
                    torso_variance =compute_covariance(torso_pose_avg, torso_pose_list)
                    result.append({
                        'name': prev_meas_filename,
                        'folder': prev_meas_folder,
                        'pose_id': prev_meas_pose_id,
                        'left_weight': prev_meas_left_weight,
                        'right_weight': prev_meas_right_weight,
                        'base_pose_avg': base_pose_avg,
                        'torso_pose_avg': torso_pose_avg,
                        'base_variance': base_variance,
                        'torso_variance': torso_variance
                    })

                # Reset meas temporary variables
                prev_meas_filename = curr_meas_filename
                prev_meas_folder = curr_meas_folder
                prev_meas_pose_id = curr_meas_pose_id
                prev_meas_left_weight = curr_meas_left_weight
                prev_meas_right_weight = curr_meas_right_weight
                base_pose_list = []
                torso_pose_list = []

                print(len(result))

            base_pose_list.append(transform_to_se3(row, "base"))
            torso_pose_list.append(transform_to_se3(row, "shoulder"))
    return result


def analyse_base_experiment():
    meas_avg_list = compute_transforms_avg_from_file("/home/earlaud/exchange/measures/merged_results.csv")

    key_field = "folder"

    # List all keys
    keys = []
    for row in meas_avg_list:
        key = row[key_field]
        if key not in keys:
            keys.append(key)

    # Initialize pivot table
    weights = {k: [] for k in keys}
    torso_rot_errs = {k: [] for k in keys}
    torso_rot_sigmas = {k: [] for k in keys}

    for meas in meas_avg_list:
        key = meas[key_field]

        def find_ref_meas(meas):
            min_weight = 1e9
            min_weight_meas = None
            for m in meas_avg_list:
                if m[key_field] != key:
                    continue
                weight = m['left_weight'] + m['right_weight']
                if weight < min_weight:
                    min_weight = weight
                    min_weight_meas = m
            return min_weight_meas

        ref_meas = find_ref_meas(meas)

        delta_weight = meas['left_weight'] + meas['right_weight'] - ref_meas['left_weight'] - ref_meas['right_weight']

        base_delta_pose = ref_meas['base_pose_avg'].inverse() * meas['base_pose_avg']
        torso_delta_pose = ref_meas['torso_pose_avg'].inverse() * meas['torso_pose_avg']

        # print(np.trace(base_delta_pose.rotation)/2. - 0.5)
        base_rot_err = np.linalg.norm(pin.rpy.matrixToRpy(base_delta_pose.rotation))
        # print(np.trace(torso_delta_pose.rotation)/2. - 0.5)
        torso_rot_err = np.linalg.norm(pin.rpy.matrixToRpy(torso_delta_pose.rotation))
        torso_rot_sigma = np.sqrt(np.linalg.norm(meas['torso_variance'][:3,:3]))

        weights[key].append(meas['left_weight'] + meas['right_weight'])
        torso_rot_errs[key].append(torso_rot_err)
        torso_rot_sigmas[key].append(torso_rot_sigma)


    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    # ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')   # Thicker major lines
    # ax.grid(True, which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Thinner minor lines
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    for key in weights.keys():
        if key != "homeboth" and key != "hightorso" and key != "maxreach-r-a":
            continue
        ax.errorbar(weights[key], torso_rot_errs[key], yerr=torso_rot_sigmas[key], fmt='o', label=f'configuration {key}', capsize=3)
        plt.draw()

    plt.ioff()  # Turn off interactive mode
    plt.ylabel("Deviation (rad)")
    plt.xlabel("Weight (kg)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    a = analyse_base_experiment()