import os
import numpy as np
from copy import deepcopy
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

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
    base_trans_errs = {k: [] for k in keys}
    base_trans_sigmas = {k: [] for k in keys}
    torso_trans_errs = {k: [] for k in keys}
    torso_trans_sigmas = {k: [] for k in keys}
    tooltip_trans_errs = {k: [] for k in keys}
    tooltip_trans_sigmas = {k: [] for k in keys}
    base_rot_errs = {k: [] for k in keys}
    base_rot_sigmas = {k: [] for k in keys}
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

        # Compute values of interest for each measure
        tot_weight = meas['left_weight'] + meas['right_weight']
        delta_weight = meas['left_weight'] + meas['right_weight'] - ref_meas['left_weight'] - ref_meas['right_weight']

        base_delta_pose = ref_meas['base_pose_avg'].inverse() * meas['base_pose_avg']
        torso_delta_pose = ref_meas['torso_pose_avg'].inverse() * meas['torso_pose_avg']

        base_trans_err = np.linalg.norm(base_delta_pose.translation)
        base_trans_sigma = np.sqrt(np.linalg.norm(meas['base_variance'][3:,3:]))

        # base_trans_err = base_delta_pose.translation[2]
        # base_trans_sigma = np.sqrt(meas['base_variance'][5,5])

        torso_trans_err = np.linalg.norm(torso_delta_pose.translation)
        torso_trans_sigma = np.sqrt(np.linalg.norm(meas['torso_variance'][3:,3:]))

        # torso_trans_err = torso_delta_pose.translation[2]
        # torso_trans_sigma = np.sqrt(meas['torso_variance'][5,5])

        if key == "homeboth":
            torso_m_tooltip = pin.XYZQUATToSE3(np.array([100.,0,200., 0,0,0,1]))
        else:
            torso_m_tooltip = pin.XYZQUATToSE3(np.array([1000.,0.,0., 0,0,0,1]))
        tooltip_delta_pose = torso_m_tooltip.inverse() * torso_delta_pose * torso_m_tooltip
        tooltip_variance = meas['torso_variance']
        tooltip_variance[3:, 3:] += (torso_m_tooltip.translation * np.transpose(torso_m_tooltip.translation)) * tooltip_variance[:3, :3]

        tooltip_trans_err = np.linalg.norm(tooltip_delta_pose.translation)
        tooltip_trans_sigma = np.sqrt(np.linalg.norm(tooltip_variance[3:,3:]))

        base_rot_err = np.linalg.norm(pin.rpy.matrixToRpy(base_delta_pose.rotation))
        base_rot_sigma = np.sqrt(np.linalg.norm(meas['base_variance'][:3,:3]))

        torso_rot_err = np.linalg.norm(pin.rpy.matrixToRpy(torso_delta_pose.rotation))
        torso_rot_sigma = np.sqrt(np.linalg.norm(meas['torso_variance'][:3,:3]))

        # Add to result list
        weights[key].append(tot_weight)

        base_trans_errs[key].append(base_trans_err)
        base_trans_sigmas[key].append(base_trans_sigma)

        torso_trans_errs[key].append(torso_trans_err)
        torso_trans_sigmas[key].append(torso_trans_sigma)

        tooltip_trans_errs[key].append(tooltip_trans_err)
        tooltip_trans_sigmas[key].append(tooltip_trans_sigma)

        base_rot_errs[key].append(base_rot_err)
        base_rot_sigmas[key].append(base_rot_sigma)

        torso_rot_errs[key].append(torso_rot_err)
        torso_rot_sigmas[key].append(torso_rot_sigma)

    # Plot results
    plt.ion()  # Turn on interactive mode

    # Rotation
    fig1, ax1 = plt.subplots()
    ax1.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')   # Thicker major lines
    ax1.grid(True, which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Thinner minor lines
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.001))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.0002))
    i = 0
    for key in weights.keys():
        if key != "homeboth" and key != "hightorso" and key != "maxreach-r-a":
            continue
        ax1.errorbar(weights[key], torso_rot_errs[key], yerr=torso_rot_sigmas[key], fmt='o', color=f"C{i}", label=f"config '{key}' : torso rotation", capsize=3)
        ax1.errorbar(weights[key], base_rot_errs[key], yerr=base_rot_sigmas[key], fmt='x', color=f"C{i}", label=f"config '{key}' : base rotation", capsize=3)
        plt.draw()
        i+=1

    ax1.set_title("Rotation error")
    ax1.set_ylabel("Deviation (rad)")
    ax1.set_xlabel("Weight (kg)")
    plt.legend()

    # Translation
    fig2, ax2 = plt.subplots()
    ax2.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')   # Thicker major lines
    ax2.grid(True, which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Thinner minor lines
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1.))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(.2))
    i = 0
    for key in weights.keys():
        if key != "homeboth" and key != "hightorso" and key != "maxreach-r-a":
            continue
        ax2.errorbar(weights[key], tooltip_trans_errs[key], yerr=tooltip_trans_sigmas[key], fmt='o', color=f"C{i}", label=f"config '{key}' : translation projected @ tooltip", capsize=3)
        # ax2.errorbar(weights[key], torso_trans_errs[key], yerr=torso_trans_sigmas[key], fmt='x', color=f"C{i}", label=f"config '{key}' : torso translation", capsize=3)
        plt.draw()
        i+=1

    ax2.set_title("Translation error")
    ax2.set_ylabel("Deviation (mm)")
    ax2.set_xlabel("Weight (kg)")

    plt.ioff()  # Turn off interactive mode
    plt.legend()
    plt.show()

if __name__ == "__main__":
    a = analyse_base_experiment()