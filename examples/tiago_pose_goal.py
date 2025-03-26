#!/usr/bin/env python3
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import tiagopro as robot


def main():
    rclpy.init()

    # Create node for this example
    node = Node("tiago_pose_goal")

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    # Create MoveIt 2 interface
    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=robot.base_link_name(),
        end_effector_name=robot.end_effector_name(),
        group_name=robot.MOVE_GROUP_ARM,
        callback_group=callback_group,
    )
    moveit2.planner_id = "RRTConnectkConfigDefault"

    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
    node.create_rate(1.0).sleep()

    # Scale down velocity and acceleration of joints (percentage of maximum)
    moveit2.max_velocity = 0.5
    moveit2.max_acceleration = 0.5

    # Get parameters
    position_1 = [0.8, 0.3, 0.25]
    position_2 = [0.3, -0.4, 0.75]

    quat_xyzw = [0.0, 0.707, 0.0, 0.707]

    for i in range(8):
        position = [ position_1[j] if (i&(1<<j) != 0) else position_2[j] for j in range(3)]
        plan = moveit2.plan(position= position, quat_xyzw = quat_xyzw)

        success = (plan is not None)

        if not success:
            print(f"{position=} not reachable !")
            continue

        # moveit2.execute(plan)
        # input("Press enter to continue...")

    # moveit2.move_to_pose(
    #     position=[0.5, 0.0, 0.5],
    #     quat_xyzw=quat_xyzw,
    #     cartesian=False
    # )

    # Note: the same functionality can be achieved by setting
    # `synchronous:=false` and `cancel_after_secs` to a negative value.
    # moveit2.wait_until_executed()

    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    main()
