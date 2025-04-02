#!/usr/bin/env python3
from threading import Thread

"""
============ TODO =================
Script pour parcourir quelques point plusieurs fois pour la repetabilité

Script pour comparer target pose et tf pose
"""

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState

from pymoveit2 import MoveIt2
from pymoveit2.robots import tiagopro as robot
from typing import Any, Tuple, Union, Optional
import os
import yaml
from time import sleep
from datetime import datetime
import pickle
import tf2_ros
import asyncio
import qtm_rt
import threading
import xml.etree.ElementTree as ET
from copy import copy

class FileLogger:
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

class MocapIF:
    def __init__(self):
        self._packet_mutex = threading.Lock()
        self._last_packet = None
        self._loop = None
        self._thread = None
        self.is_ready = False

        # Start the event loop in a separate thread
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

    def _run_event_loop(self):
        """Run the event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Schedule the setup coroutine
        asyncio.run_coroutine_threadsafe(self._qtm_setup(), self._loop)

        # Run the event loop
        self._loop.run_forever()

    async def _qtm_setup(self):
        """Setup QTM connection."""
        # create connection
        self.connection = await qtm_rt.connect("192.168.11.60")
        if self.connection is None:
            raise ConnectionError("Failed to connect to QTM")

        # Start streaming
        await self.connection.stream_frames(
            components=["6d", "timecode"],
            on_packet=self._on_packet
        )

        # Get Body name to index mapping
        xml_string = await self.connection.get_parameters(parameters=["6d"])
        xml = ET.fromstring(xml_string)
        self.body_to_index = {}

        for index, body in enumerate(xml.findall("*/Body/Name")):
            self.body_to_index[body.text.strip()] = index

        self.is_ready = True

    def _on_packet(self, packet):
        timestamp = packet.timestamp
        self._packet_mutex.acquire()
        try:
            self._last_packet = packet
        finally:
            self._packet_mutex.release()

    def get_poses(self, bodies):
        timecode = None
        framenumber = None
        poses = []
        self._packet_mutex.acquire()
        if self._last_packet:
            framenumber = self._last_packet.framenumber
            timecode = self._last_packet.get_timecode()
            for body in bodies:
                info, bodies = self._last_packet.get_6d()
                pose_obj = copy(bodies[self.body_to_index[body_name]])
                pose = [pose[0].x, pose[0].y, pose[0].y], pose[1].matrix
                poses.append(pose)
        self._packet_mutex.release()
        return framenumber, timecode, poses

class MeasureNode(Node):
    def __init__(self, mocap_if):
        super().__init__("MeasureNode")

        # Mocap
        self.mocap_if = mocap_if

        # Tf setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Logger
        self.logger = FileLogger("~/exchange/measures")
        self.end_comment = False

    def _finish(self):
        if not self.end_comment:
            end_comment = input("Any last comment? ")
            self.logger.add_metadata("end_comment", end_comment)
            self.end_comment = True

    def __del__(self):
        self._finish()

    def _record_measures(self, logger, duration_s=2.0):
        tf_positions = []
        tf_orientations = []
        joint_positions = []
        joint_velocities = []
        joint_efforts = []
        mocap_framenumbers = []
        mocap_timecodes = []
        mocap_positions = []
        mocap_rotmats = []

        deadline = self.get_clock().now() + rclpy.time.Duration(seconds=duration_s)
        while(rclpy.ok() and self.get_clock().now() <= deadline):
            # if(not self.moveit2.new_joint_state_available):
            #     continue
            # self.moveit2.reset_new_joint_state_checker()

            transform_msg = self.tf_buffer.lookup_transform(robot.end_effector_name(), robot.base_link_name(), rclpy.time.Time())
            tf_position = transform_msg.transform.translation.x, transform_msg.transform.translation.y, transform_msg.transform.translation.z
            tf_orientation = transform_msg.transform.rotation.x, transform_msg.transform.rotation.y, transform_msg.transform.rotation.z, transform_msg.transform.rotation.w
            # joint_states = self.moveit2.joint_state
            # mocap_data = self.mocap_if.get_body_pose("support")

            tf_positions.append(tf_position)
            tf_orientations.append(tf_orientation)
            # joint_positions.append(joint_states.position.tolist())
            # joint_velocities.append(joint_states.velocity.tolist())
            # joint_efforts.append(joint_states.effort.tolist())
            # mocap_framenumbers.append(mocap_data[0])
            # mocap_timecodes.append(mocap_data[1])
            # mocap_positions.append(mocap_data[2])
            # mocap_rotmats.append(mocap_data[3])


        logger.add_meas("tf_positions", tf_positions)
        logger.add_meas("tf_orientations", tf_orientations)
        logger.add_meas("joint_positions", joint_positions)
        logger.add_meas("joint_velocities", joint_velocities)
        logger.add_meas("joint_efforts", joint_efforts)
        logger.add_meas("mocap_framenumbers", mocap_framenumbers)
        logger.add_meas("mocap_timecodes", mocap_timecodes)
        logger.add_meas("mocap_positions", mocap_positions)
        logger.add_meas("mocap_rotmats", mocap_rotmats)
        print(f"Logged {len(tf_positions)} points in {duration_s} s")

    def run(self,):
        start_comment = input("Please comment this experiment: ")
        self.logger.add_metadata("start_comment", start_comment)
        # self.logger.add_metadata("joint_names", self.moveit2.joint_state.name)

        meas_id = 0
        while rclpy.ok():
            input("Press enter to record new points...")
            self.logger.clear()
            self._record_measures(self.logger)
            self.logger.save(meas_id)
            meas_id += 1

def main():
    rclpy.init()

    # Mocap
    mocap_if = None #MocapIF()
    # print("Wainting for mocap...")
    # while not mocap_if.is_ready:
    #     pass
    # print("Done")

    meas_node = MeasureNode(mocap_if)

    # Run user script in another thread
    script_thread = Thread(target=meas_node.run, daemon=True, args=())
    script_thread.start()

    rclpy.spin(meas_node)

    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    main()
