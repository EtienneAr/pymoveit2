#!/usr/bin/env python3
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from visualization_msgs.msg import Marker

from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import tiagopro as robot
from typing import Any, Tuple, Union, Optional
import os
import yaml
from time import sleep
from datetime import datetime
import pickle
import tf2_ros

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

class MeasuresLogger:
    def __init__(self, path: Optional[str] = "."):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dirpath = os.path.expanduser(os.path.join(path, timestamp))
        self.dirpath = dirpath
        os.makedirs(self.dirpath)
        self.clear()

    def clear(self):
        self._buffer = {}

    def add_metadata(self, key: str, value: str):
        filepath = os.path.join(self.dirpath, "metadata.yaml")
        try:
            with open(filepath, "r") as f:
                metadata = yaml.safe_load(f)
                if(not metadata):
                    raise FileNotFoundError() # Raise an error to jump to the catch and initialize metadata properly
        except FileNotFoundError as e:
            metadata = {}
        with open(filepath, "w+") as f:
            metadata.update({key: value})
            yaml.dump(metadata, f)

    def add_meas(self, measure_name: Union[int, str], obj: Any):
        self._buffer.update({measure_name: obj})

    def save(self, measure_id: str):
        filepath = os.path.join(self.dirpath, f"{measure_id}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self._buffer, f)
        self.clear()

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
        self.moveit2.add_collision_cylinder("mocap_markers",
            0.07, #height
            0.03, # radius
            position = [0.035,0.,0.],
            quat_xyzw = [0.0, 0.707, 0.0, 0.707],
            frame_id = "arm_left_tool_link")
        self.moveit2.attach_collision_object("mocap_markers", "arm_left_tool_link", ["gripper_left_base_link", "arm_left_7_link", "arm_left_6_link"])

        # Tf setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        # Logger
        self.logger = MeasuresLogger("~/exchange/measures")
        self.end_comment = False

    def finish(self):
        if not self.end_comment:
            end_comment = input("Any last command? ")
            self.logger.add_metadata("end_comment", end_comment)
            self.end_comment = True

    def __del__(self):
        self.finish()

    def run(self):
        # Get parameters
        position_ref = [0.7, 0.2, 0.95]
        sampling_box = [0.4, 0.4, 0.6]
        subdivide = [4, 4, 4]

        grid = PointGrid(position_ref, sampling_box, subdivide)

        # User comment
        start_comment = input("Please comment this experiment: ")
        self.logger.add_metadata("start_comment", start_comment)

        self.logger.add_metadata("joint_names", self.moveit2.joint_state.name)

        quat_xyzw = [0.0, 0.707, 0.0, 0.707]

        meas_nb = 0
        for i, position in enumerate(grid.points):
            plan = self.moveit2.plan(position= position, quat_xyzw = quat_xyzw)

            success = (plan is not None)

            # Diplay marker in RViz
            self.reachability_marker.header.stamp = self.node.get_clock().now().to_msg()
            self.reachability_marker.id = i
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
                continue

            # Move to pose
            self.moveit2.execute(plan)
            success = self.moveit2.wait_until_executed()

            if not success:
                print(f"{position=} execution failed !")
                continue

            # Log only if successful
            self.logger.clear()

            self.logger.add_meas("target_index", i)
            self.logger.add_meas("target_position", position)
            self.logger.add_meas("target_orientation", quat_xyzw)

            transform_msg = self.tf_buffer.lookup_transform(robot.end_effector_name(), robot.base_link_name(), rclpy.time.Time())
            tf_position = transform_msg.transform.translation.x, transform_msg.transform.translation.y, transform_msg.transform.translation.z
            tf_orientation = transform_msg.transform.rotation.x, transform_msg.transform.rotation.y, transform_msg.transform.rotation.z, transform_msg.transform.rotation.w
            joint_states = self.moveit2.joint_state

            self.logger.add_meas("tf_position", tf_position)
            self.logger.add_meas("tf_orientation", tf_orientation)
            self.logger.add_meas("joint_position", joint_states.position.tolist())
            self.logger.add_meas("joint_velocity", joint_states.velocity.tolist())
            self.logger.add_meas("joint_effort", joint_states.effort.tolist())

            self.logger.save(meas_nb)
            meas_nb +=1

        self.finish()

def main():
    rclpy.init()

    # Create node for this example
    node = Node("tiago_pose_goal")

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    experiment =  Experiment(node, callback_group)

    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
    node.create_rate(1.0).sleep()

    experiment.run()

    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    main()
