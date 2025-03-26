from typing import List

MOVE_GROUP_ARM: str = "arm_left"
MOVE_GROUP_GRIPPER: str = "gripper_left"

OPEN_GRIPPER_JOINT_POSITIONS: List[float] = [1.0]
CLOSED_GRIPPER_JOINT_POSITIONS: List[float] = [0.0]


def joint_names(prefix: str = "arm_left_") -> List[str]:
    return [
        prefix + "1_joint",
        prefix + "2_joint",
        prefix + "3_joint",
        prefix + "4_joint",
        prefix + "5_joint",
        prefix + "6_joint",
        prefix + "7_joint",
    ]


def base_link_name() -> str:
    return "base_link"


def end_effector_name(prefix: str = "arm_left_") -> str:
    return prefix + "tool_link"


def gripper_joint_names(prefix: str = "gripper_left_") -> List[str]:
    return [
        prefix + "finger_joint",
    ]
