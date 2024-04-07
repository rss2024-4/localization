import math
import sys

from geometry_msgs.msg import TransformStamped

import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q

class StaticTfCamPublisher(Node):
    def __init__(self):
        super().__init__('static_tf_cam_publisher')
        
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        
        # Publish static transforms once at startup
        self.make_transforms(["base_link", 'laser', 0.0, 0, 0, 0, 0, 0])
        # self.make_transforms(["left_cam", 'right_cam', 0.1, 0, 0, 0, 0, 0])
        # self.make_transforms(right_to_left)

    def make_transforms(self, transformation):
        ts = []
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = transformation[0]
        t.child_frame_id = transformation[1]

        t.transform.translation.x = float(transformation[2])
        t.transform.translation.y = float(transformation[3])
        t.transform.translation.z = float(transformation[4])
        quat = quaternion_from_euler(
            float(transformation[5]), float(transformation[6]), float(transformation[7]))
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        ts.append(t)

        self.tf_static_broadcaster.sendTransform(ts)

def main(args=None):
    rclpy.init(args=args)

    # left_to_world = ["base_link_gt", 'left_cam', -0.05, 0, 0, 0, 0, 0]
    # right_to_left = ["left_cam", 'right_cam', 0.1, 0, 0, 0, 0, 0]

    pub = StaticTfCamPublisher()
    # right_static_publisher = StaticTfCamPublisher(right_to_left)

    rclpy.spin(pub)
    # rclpy.spin(right_static_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pub.destroy_node()
    # right_static_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
