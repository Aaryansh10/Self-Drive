#!/usr/bin/env python3
"""Exercise a latched map older than a late-starting node's dynamic TF buffer."""
import os
import subprocess
import time
from ament_index_python.packages import get_package_prefix
import rclpy
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from tf2_ros.transform_broadcaster import TransformBroadcaster


def main():
    rclpy.init()
    node = rclpy.create_node('centerline_tf_startup_test')
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pubs = [node.create_publisher(OccupancyGrid, '/lane_map/' + c + '_skeleton', qos)
            for c in ['white', 'yellow']]
    results = {}
    subs = [node.create_subscription(Path, '/path/' + side + '_centerline',
            lambda m, side=side: results.update({side: m}), 10) for side in ['left', 'right']]
    tf = TransformBroadcaster(node)

    def broadcast():
        msg = TransformStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'link_base'
        msg.transform.rotation.w = 1.
        tf.sendTransform(msg)

    def maps(stamp):
        output = []
        for color in ['white', 'yellow']:
            m = OccupancyGrid()
            m.header.frame_id = 'odom'
            m.header.stamp = stamp
            m.info.resolution = .1
            m.info.width = m.info.height = 120
            m.info.origin.position.x = -2.
            m.info.origin.position.y = -6.
            m.info.origin.orientation.w = 1.
            m.data = [0] * (120 * 120)
            for x in range(20, 100):
                for y in ([23, 97] if color == 'white' else [60]):
                    m.data[y * 120 + x] = 100
            output.append(m)
        return output

    for _ in range(4):
        broadcast()
        rclpy.spin_once(node, timeout_sec=.03)
    old = (node.get_clock().now() - Duration(seconds=.3)).to_msg()
    for pub, msg in zip(pubs, maps(old)):
        pub.publish(msg)
    process = subprocess.Popen([os.path.join(get_package_prefix('path_generation_cpp'), 'lib',
                                            'path_generation_cpp', 'centerline_node')],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            broadcast()
            rclpy.spin_once(node, timeout_sec=.05)
            if all(side in results and results[side].poses for side in ['left', 'right']):
                break
        else:
            raise AssertionError('Late subscriber failed to generate paths from latched map')
        for msg in results.values():
            assert (msg.header.stamp.sec, msg.header.stamp.nanosec) > (old.sec, old.nanosec), \
                'Path must use the TF pose timestamp, not the older map stamp'
        stale = (node.get_clock().now() - Duration(seconds=5)).to_msg()
        for pub, msg in zip(pubs, maps(stale)):
            pub.publish(msg)
        results.clear()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            broadcast()
            rclpy.spin_once(node, timeout_sec=.05)
            if len(results) == 2 and all(not msg.poses for msg in results.values()):
                break
        else:
            raise AssertionError('A map outside the allowed TF skew was not rejected')
        print('PASS: late-start latched map uses current TF; stale map is rejected')
    finally:
        process.terminate()
        process.wait(timeout=5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
