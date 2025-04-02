#!/usr/bin/env python3
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from visualization_msgs.msg import Marker

from pymoveit2 import MoveIt2
from pymoveit2.robots import tiagopro as robot
from typing import Any, Tuple, Union, Optional
from time import sleep
import asyncio

class PointGrid:
    def __init__(self, center: Tuple[float, float, float], volume = Tuple[float, float, float], subdivisions = Tuple[int, int, int]):
        self.center = center
        self.volume = volume
        self.subdivisions = subdivisions
        self._populate_points()

    def _populate_points(self):
        n_points = 1
        divisors = []
        for div in self.subdivisions:
            divisors.append(n_points)
            n_points *= div
        self.n_points = n_points
        self.divisors = divisors

        self.points = []
        for i in range(self.n_points):
            position = [0, 0, 0,]
            for j in range(3):
                substep = (i // self.divisors[j]) % self.subdivisions[j]
                pos_offset = self.volume[j] * (2.0 * substep / self.subdivisions[j] - 1.0)
                position[j] = self.center[j] + pos_offset
            self.points.append(position)

    def get(self, i: int):
        return self.points[i]

    def get(self, indexes: Tuple[int, int, int]):
        index = 0
        for j in range(3):
            assert(indexes[j] < self.subdivisions[j])
            index += indexes[j] * self.divisors[j]
        return self.get(index)

class Experiment:
    def __init__(self, node, callback_group):
        self.node = node

        # Debugging markers
        self.reachability_publisher = node.create_publisher(Marker, 'reachability', 10)
        self.reachability_marker = Marker()
        self.reachability_marker.header.frame_id = robot.base_link_name()
        self.reachability_marker.ns = 'spheres'
        self.reachability_marker.type = Marker.SPHERE
        self.reachability_marker.action = Marker.ADD
        self.reachability_marker.pose.orientation.w = 1.0
        self.reachability_marker.scale.x = 0.05
        self.reachability_marker.scale.y = 0.05
        self.reachability_marker.scale.z = 0.05
        self.reachability_marker.color.a = 1.0  # Fully opaque
        self.reachability_marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()

        # Create MoveIt 2 interface
        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )
        self.moveit2.planner_id = "RRTConnectkConfigDefault"

        # Scale down velocity and acceleration of joints (percentage of maximum)
        self.moveit2.max_velocity = 0.5
        self.moveit2.max_acceleration = 0.5
        self.moveit2.allowed_planning_time = 2.0

        # Add collision objects
        cyl_height = 0.105
        cyl_rad = 0.105
        self.moveit2.add_collision_cylinder("mocap_markers",
            cyl_height, #height
            cyl_rad, # radius
            position = [0.0,0., cyl_height/2. - 0.0125],
            quat_xyzw = [0.0, 0.0, 0.0, 1.0],
            frame_id = "arm_left_7_link")
        self.moveit2.attach_collision_object("mocap_markers", "arm_left_7_link", ["gripper_left_base_link", "arm_left_tool_link", "arm_left_7_link", "arm_left_6_link"])

    def go_to(self, position, quat_xyzw, *, execute=False, debug_id = 0):
        plan = self.moveit2.plan(position= position, quat_xyzw = quat_xyzw)
        success = (plan is not None)

        # Diplay marker in RViz
        self.reachability_marker.header.stamp = self.node.get_clock().now().to_msg()
        self.reachability_marker.id = debug_id
        self.reachability_marker.pose.position.x = position[0]
        self.reachability_marker.pose.position.y = position[1]
        self.reachability_marker.pose.position.z = position[2]
        self.reachability_marker.color.r = 1.0 if not success else 0.0
        self.reachability_marker.color.g = 1.0 if success else 0.0
        self.reachability_marker.color.b = 0.0
        self.reachability_publisher.publish(self.reachability_marker)
        sleep(0.1)

        if not success:
            print(f"{position=} not reachable !")
            return False

        if not execute:
            # Do not execute trajectory, terminate here
            return True

        # Move to pose
        self.moveit2.execute(plan)
        success = self.moveit2.wait_until_executed()

        if not success:
            print(f"{position=} execution failed !")
            # Diplay marker in RViz
            self.reachability_marker.header.stamp = self.node.get_clock().now().to_msg()
            self.reachability_marker.id = debug_id
            self.reachability_marker.pose.position.x = position[0]
            self.reachability_marker.pose.position.y = position[1]
            self.reachability_marker.pose.position.z = position[2]
            self.reachability_marker.color.r = 1.0
            self.reachability_marker.color.g = 1.0
            self.reachability_marker.color.b = 0.0
            self.reachability_publisher.publish(self.reachability_marker)
            sleep(0.1)
            return False

        return True

    def run_charge(self, point_id):
        # Get parameters
        position_ref = [0.7, 0.2, 0.95]
        sampling_box = [0.4, 0.4, 0.6]
        subdivide = [4, 4, 4]

        grid = PointGrid(position_ref, sampling_box, subdivide)

        weight_list = [0,1,2,3,4,5,-1]

        quat_xyzw = [0.0, 0.707, 0.0, 0.707]

        # Go to position
        position = grid.points[point_id]
        self.go_to(position, quat_xyzw, execute=True, debug_id = 0)

        for weight in weight_list:
            input(f"Put {weight} weights on the robots and press enter...")

    def run(self):
        # Get parameters
        position_ref = [0.7, 0.2, 0.95]
        sampling_box = [0.4, 0.4, 0.6]
        subdivide = [4, 4, 4]

        grid = PointGrid(position_ref, sampling_box, subdivide)

        quat_xyzw = [0.0, 0.707, 0.0, 0.707]

        for i, position in enumerate(grid.points):
            input("Press enter to go to next point...")
            # Go to position
            self.go_to(position, quat_xyzw, execute=True, debug_id = i)

def main():
    rclpy.init()

    # Create node for this example
    node = Node("tiago_move_node")

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    experiment =  Experiment(node, callback_group)

    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
    node.create_rate(1.0).sleep()

    # experiment.run()
    experiment.run_charge(31)

    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    main()
