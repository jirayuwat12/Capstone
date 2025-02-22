import unittest
import warnings
from collections import deque

import numpy as np

from capstone_utils.relative_angle_conversion import position_to_relative_angle, relative_angle_to_position
from capstone_utils.skeleton_utils.progressive_trans_model import (
    JOINT_TO_CHILD_JOINTS_INDEX,
    JOINT_TO_PREV_JOINT_INDEX,
    ROOT_JOINT,
    SKELETON_MODEL,
)


class TestRelativeAngleConversion(unittest.TestCase):
    def setUp(self):
        # Create random joint positions
        self.random_joints = np.random.rand(50, 3)
        self.random_joints = self.random_joints - self.random_joints[0]
        # Create all straight joints
        self.y_straigh_joints = np.zeros((50, 3))
        self.x_straigh_joints = np.zeros((50, 3))
        self.z_straigh_joints = np.zeros((50, 3))
        queue = deque()
        queue.append(ROOT_JOINT)
        while queue:
            parent_joint = queue.popleft()
            if parent_joint not in JOINT_TO_CHILD_JOINTS_INDEX:
                continue
            for child_joint in JOINT_TO_CHILD_JOINTS_INDEX[parent_joint]:
                self.y_straigh_joints[child_joint, 1] = self.y_straigh_joints[parent_joint, 1] + 1
                self.x_straigh_joints[child_joint, 0] = self.x_straigh_joints[parent_joint, 0] + 1
                self.z_straigh_joints[child_joint, 2] = self.z_straigh_joints[parent_joint, 2] + 1
                queue.append(child_joint)
        # Create exception joint positions
        self.exception_joints = [0, 4, 7]
        warnings.warn(f"This test exception is for bone index {self.exception_joints}")

    def is_same_joint(self, joint1, joint2):
        """Check is same joint position with exception"""
        for exception_joint in self.exception_joints:
            joint1[exception_joint] = joint2[exception_joint]

        if not np.allclose(joint1, joint2):
            for bone_index in range(50):
                if not np.allclose(joint1[bone_index], joint2[bone_index]):
                    print(f"Bone index {bone_index} is not the same")
                    print("original\t:", joint1[bone_index])
                    print("positional\t:", joint2[bone_index])
                    bone_path = [bone_index]
                    current_bone = bone_index
                    while current_bone != 0:
                        if current_bone not in JOINT_TO_PREV_JOINT_INDEX:
                            break
                        prev_bone = JOINT_TO_PREV_JOINT_INDEX[current_bone]
                        bone_path.insert(0, prev_bone)
                    print("Bone path:", bone_path)
            self.assertTrue(False)

    def test_ptr_workable(self):
        """Test if the function works"""
        relative_angle = position_to_relative_angle(self.random_joints, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
        self.assertTrue(relative_angle is not None)

    def test_rtp_workable(self):
        """Test if the function works"""
        position = relative_angle_to_position(
            self.random_joints, SKELETON_MODEL, JOINT_TO_PREV_JOINT_INDEX, JOINT_TO_CHILD_JOINTS_INDEX, ROOT_JOINT
        )
        self.assertTrue(position is not None)

    def test_return_same(self):
        """Test if the function returns the same value"""
        relative_angle = position_to_relative_angle(self.random_joints, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
        position = relative_angle_to_position(
            relative_angle, SKELETON_MODEL, JOINT_TO_PREV_JOINT_INDEX, JOINT_TO_CHILD_JOINTS_INDEX, ROOT_JOINT
        )
        self.is_same_joint(self.random_joints, position)

    def test_ptr_correct_x_angle(self):
        """Test if the function returns the correct x angle"""
        relative_angle = position_to_relative_angle(self.x_straigh_joints, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
        for bone_index in range(50):
            if bone_index in self.exception_joints:
                self.assertTrue(np.allclose(relative_angle[bone_index].sum(), 0))
            else:
                # Check the length is 1
                self.assertTrue(np.allclose(relative_angle[bone_index][0], 1))
            if bone_index == 1:
                # Check the angle is 90 degree
                self.assertTrue(np.allclose(relative_angle[bone_index][1], np.deg2rad(90)))
            else:
                # Check the angle is 0 degree
                self.assertTrue(np.allclose(relative_angle[bone_index][1], 0))
            # Check the angle is 0 degree
            self.assertTrue(np.allclose(relative_angle[bone_index][2], 0))

    def test_ptr_correct_y_angle(self):
        """Test if the function returns the correct y angle"""
        relative_angle = position_to_relative_angle(self.y_straigh_joints, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
        for bone_index in range(50):
            if bone_index in self.exception_joints:
                self.assertTrue(np.allclose(relative_angle[bone_index].sum(), 0))
            else:
                # Check the length is 1
                self.assertTrue(np.allclose(relative_angle[bone_index][0], 1))
            if bone_index == 1:
                # Check the angle is 90 degree
                self.assertTrue(np.allclose(relative_angle[bone_index][1], np.deg2rad(90)))
                self.assertTrue(np.allclose(relative_angle[bone_index][2], np.deg2rad(90)))
            else:
                # Check the angle is 0 degree
                self.assertTrue(np.allclose(relative_angle[bone_index][1], 0))
                self.assertTrue(np.allclose(relative_angle[bone_index][2], 0))

    def test_ptr_correct_z_angle(self):
        """Test if the function returns the correct z angle"""
        relative_angle = position_to_relative_angle(self.z_straigh_joints, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
        for bone_index in range(50):
            if bone_index in self.exception_joints:
                self.assertAlmostEqual(relative_angle[bone_index].sum(), 0)
            else:
                # Check the length is 1
                self.assertAlmostEqual(relative_angle[bone_index][0], 1)
                # Check the angle is 0 degree
                self.assertAlmostEqual(relative_angle[bone_index][1], 0)
                self.assertAlmostEqual(relative_angle[bone_index][2], 0)
