#!/usr/bin/env python3
import rclpy
import time
import os
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Pose
from pyquaternion import Quaternion as PyQuaternion
from threading import Thread
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2, MoveIt2State
from .gen3lite_pymoveit2 import Gen3LiteArm, Gen3LiteGripper

script_dir = os.path.dirname(os.path.realpath(__file__))
green_cube_pick_data = np.loadtxt(os.path.join(script_dir, 'Pick.csv'), delimiter=',')
green_cube_put_data = np.loadtxt(os.path.join(script_dir, 'Put.csv'), delimiter=',')

class Gen3LiteArm:
    def __init__(self):
        self.node = Node("gen3_lite_arm")
        self.callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=[
                'joint_1', 'joint_2', 'joint_3',
                'joint_4', 'joint_5', 'joint_6',
                'end_effector_link'
            ],
            base_link_name='base_link',
            end_effector_name='end_effector_link',
            group_name='arm',
            callback_group=self.callback_group,
        )
        self.executor = rclpy.executors.MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        self.node.create_rate(1.0).sleep()

        self.moveit2.pipeline_id = 'pilz_industrial_motion_planner'
        self.moveit2.planner_id = 'LIN'
        self.moveit2.allowed_planning_time = 5.0
        self.moveit2.num_planning_attempts = 10
        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1
        self.moveit2.cartesian_jump_threshold = 0.0

    def inverse_kinematic_movement(self, target_pose, cartesian=False):
        self.node.get_logger().info(
            f"Moving to pose: {target_pose.position} {target_pose.orientation} with cartesian={cartesian}"
        )
        self.moveit2.move_to_pose(
            pose=target_pose,
            cartesian=cartesian,
            cartesian_max_step=0.0025,
            cartesian_fraction_threshold=0.0,
        )
        rate = self.node.create_rate(10)
        while self.moveit2.query_state() != MoveIt2State.EXECUTING:
            rate.sleep()
        future = self.moveit2.get_execution_future()
        while not future.done():
            rate.sleep()

def slerp(data, qStart, qEnd, arm):
    qList = []
    for q in PyQuaternion.intermediates(qStart, qEnd, len(data) - 2, include_endpoints=True):
        qList.append(q.elements)
    for i in range(len(data)):
        pose = Pose()
        pose.position.x = data[i][0]
        pose.position.y = data[i][1]
        pose.position.z = data[i][2]
        pose.orientation.x = qList[i][0]
        pose.orientation.y = qList[i][1]
        pose.orientation.z = qList[i][2]
        pose.orientation.w = qList[i][3]
        arm.inverse_kinematic_movement(pose)
        print(f"Reached control point {i}")

def pick_and_place(pick_data, put_data, arm, gripper, qStart, qPick, qPut):
    print("Moving to pick position (via SLERP)")
    slerp(pick_data, qStart, qPick, arm)

    print("Closing gripper to grasp")
    gripper.move_to_position(0.7)
    time.sleep(0.5)

    print("Moving to put position (via SLERP)")
    slerp(put_data, qPick, qPut, arm)

    print("Opening gripper to release")
    gripper.move_to_position(0.0)

def main(args=None):
    rclpy.init(args=args)
    print("Task Start")
    arm = Gen3LiteArm()
    gripper = Gen3LiteGripper()
    qStartGreenCube = PyQuaternion(array=np.array([1, 0, 0, 0]))
    qPickGreenCube = PyQuaternion(array=np.array([1, 0, 0, 0]))
    qPutGreenCube = PyQuaternion(array=np.array([1, 0, 0, 0]))

    pick_and_place(green_cube_pick_data, green_cube_put_data, arm, gripper,
                   qStartGreenCube, qPickGreenCube, qPutGreenCube)

    print("Task Finished")
    gripper.shutdown()
    arm.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

