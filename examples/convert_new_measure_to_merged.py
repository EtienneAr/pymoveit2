import os
import csv
import re
from pathlib import Path

input_dir = "/media/earlaud/BIGGYPRINTY/torso-stiff-v2"
output_csv = "/home/earlaud/exchange/measures/output_merged_thomas_06-06-25.csv"
frequency = 100  # fallback

header = [
    "Frame", "Time", "empty 0",
    "base X", "base Y", "base Z", "base Roll", "base Pitch", "base Yaw", "base Residual",
    "base Rot[0]", "base Rot[1]", "base Rot[2]", "base Rot[3]", "base Rot[4]", "base Rot[5]", "base Rot[6]", "base Rot[7]", "base Rot[8]",
    "empty 1",
    "shoulder X", "shoulder Y", "shoulder Z", "shoulder Roll", "shoulder Pitch", "shoulder Yaw", "shoulder Residual",
    "shoulder Rot[0]", "shoulder Rot[1]", "shoulder Rot[2]", "shoulder Rot[3]", "shoulder Rot[4]", "shoulder Rot[5]", "shoulder Rot[6]", "shoulder Rot[7]", "shoulder Rot[8]",
    "empty 2",
    "Folder", "Name", "PoseID", "Left_bags", "Right_bags", "Left_weight", "Right_weight"
]

def parse_filename(filename):
    name_parts = filename.stem.split("_")
    if len(name_parts) >= 3:
        pose_id = name_parts[0]
        left_w = re.search(r"(\d+)L", name_parts[1])
        right_w = re.search(r"(\d+)R", name_parts[2])
        return pose_id, int(left_w.group(1)) if left_w else 0, int(right_w.group(1)) if right_w else 0
    return "unknown", 0, 0

def find_body_indices(header_line, target_bodies):
    indices = {}
    fields = header_line.strip().split("\t")
    for i, token in enumerate(fields):
        for body in target_bodies:
            if token.strip() == f"{body} X":
                indices[body] = i
    return indices

def extract_data(tsv_path):
    rows = []
    with open(tsv_path, 'r') as f:
        lines = f.readlines()

    # Metadata extraction
    meta = {}
    for line in lines:
        if "\t" not in line.strip():
            continue
        parts = line.strip().split("\t")
        key = parts[0].strip()
        meta[key] = parts[1:] if len(parts) > 1 else []

        if key == "FREQUENCY":
            try:
                global frequency
                frequency = int(parts[1])
            except:
                pass

    # Find header line and data start
    header_line_index = None
    for i, line in enumerate(lines):
        if "X" in line and "base" in line:
            header_line_index = i
            break

    body_names = ["base", "Torso"]  # adjust "tool_link" as "shoulder"
    body_index_map = find_body_indices(lines[header_line_index], body_names)

    # Data starts right after the header line
    data_lines = lines[header_line_index + 1:]
    frame_idx = 1

    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue

        time = (frame_idx - 1) / frequency

        base_start = body_index_map["base"]
        base_data = parts[base_start:base_start+16]

        shoulder_start = body_index_map["Torso"]
        shoulder_data = parts[shoulder_start:shoulder_start+16]

        folder_name = tsv_path.parent.name
        pose_id, left_w, right_w = parse_filename(tsv_path)

        row = [frame_idx, f"{time:.2f}", ""] + \
              base_data + [""] + \
              shoulder_data + [""] + \
              [pose_id, tsv_path.name, pose_id, 0, 0, left_w, right_w]
        rows.append(row)
        frame_idx += 1

    return rows

# Main logic
all_rows = []

for file in Path(input_dir).glob("*.tsv"):
    data_rows = extract_data(file)
    all_rows.extend(data_rows)

# Write output CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print(f"CSV successfully saved to {output_csv}")