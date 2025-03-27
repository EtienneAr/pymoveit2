#!/usr/bin/env python3
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from visualization_msgs.msg import Marker

from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import tiagopro as robot
from time import sleep

def main():
    rclpy.init()

    # Create node for this example
    node = Node("tiago_pose_goal")

    # Debugging markers
    reachability_publisher = node.create_publisher(Marker, 'reachability', 10)
    reachability_marker = Marker()
    reachability_marker.header.frame_id = robot.base_link_name()
    reachability_marker.ns = 'spheres'
    reachability_marker.type = Marker.SPHERE
    reachability_marker.action = Marker.ADD
    reachability_marker.pose.orientation.w = 1.0
    reachability_marker.scale.x = 0.05
    reachability_marker.scale.y = 0.05
    reachability_marker.scale.z = 0.05
    reachability_marker.color.a = 1.0  # Fully opaque
    reachability_marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()

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
    position_ref = [0.65, 0.2, 0.95]
    sampling_box = [0.3, 0.4, 0.6]
    subdivide = 4

    quat_xyzw = [0.0, 0.707, 0.0, 0.707]

    for i in range(subdivide**3):
        current_position = [0, 0, 0,]
        substeps = []
        for j in range(3):
            substep = (i // (subdivide**j)) % subdivide
            pos_offset = 2.0 * sampling_box[j] * substep / subdivide - sampling_box[j]
            current_position[j] = position_ref[j] + pos_offset
            substeps.append(substep)
        print(substeps)
        plan = moveit2.plan(position= current_position, quat_xyzw = quat_xyzw)

        success = (plan is not None)

        # Diplay marker in RViz
        reachability_marker.header.stamp = node.get_clock().now().to_msg()
        reachability_marker.id = i
        reachability_marker.pose.position.x = current_position[0]
        reachability_marker.pose.position.y = current_position[1]
        reachability_marker.pose.position.z = current_position[2]
        reachability_marker.color.r = 1.0 if not success else 0.0
        reachability_marker.color.g = 1.0 if success else 0.0
        reachability_marker.color.b = 0.0
        reachability_publisher.publish(reachability_marker)
        sleep(0.1)

        if not success:
            print(f"{current_position=} not reachable !")
            continue

        moveit2.execute(plan)
        moveit2.wait_until_executed()

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
