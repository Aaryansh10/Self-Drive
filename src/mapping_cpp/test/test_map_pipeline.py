#!/usr/bin/env python3
"""ROS smoke test: synthetic cleaned clouds -> maps -> skeleton maps -> paths.
Run after sourcing the workspace. Uses a separate ROS domain via the caller.
"""
import os
import subprocess
import time

import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def main():
    rclpy.init()
    node = rclpy.create_node('map_pipeline_test')
    processes = []
    logs = []
    try:
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        latest = {}
        subscriptions = []
        topics = ['/lane_map/white', '/lane_map/yellow', '/lane_map/white_skeleton',
                  '/lane_map/yellow_skeleton', '/path/left_centerline', '/path/right_centerline']
        for topic in topics:
            is_path = topic.startswith('/path/')
            subscriptions.append(node.create_subscription(
                Path if is_path else OccupancyGrid, topic,
                lambda msg, topic=topic: latest.update({topic: msg}), 10 if is_path else qos))
        publishers = [node.create_publisher(point_cloud2.PointCloud2, topic, qos_profile_sensor_data)
                      for topic in ['/pc_whitelane', '/pc_yellowlane']]
        broadcaster = StaticTransformBroadcaster(node)
        transform = TransformStamped()
        transform.header.stamp = node.get_clock().now().to_msg()
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'link_base'
        transform.transform.translation.x = 40.0
        transform.transform.translation.y = 30.0
        transform.transform.rotation.z = 2 ** -0.5
        transform.transform.rotation.w = 2 ** -0.5
        broadcaster.sendTransform(transform)
        for package, executable in [('mapping_cpp', 'mapping_node'),
                                    ('mapping_cpp', 'map_skeleton_node'),
                                    ('path_generation_cpp', 'centerline_node')]:
            log = open(f'/tmp/{executable}_smoke.log', 'w')
            logs.append(log)
            command = [os.path.join(get_package_prefix(package), 'lib', package, executable)]
            if executable == 'mapping_node':
                command += ['--ros-args', '-p', 'publish_every:=1']
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))

        def spin_until(predicate, seconds=10):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if predicate():
                    return
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('Pipeline node exited; inspect /tmp/*_smoke.log')
            raise AssertionError('Timed out waiting for pipeline output')

        spin_until(lambda: all(p.get_subscription_count() for p in publishers))
        white = [(x/20, side*3.7 + offset/20, 0.0)
                 for x in range(24, 151) for side in [-1, 1] for offset in [-2, -1, 0, 1, 2]]
        yellow = [(x/20, offset/20, 0.0) for x in range(24, 151) for offset in [-2, -1, 0, 1, 2]]
        for _ in range(18):
            header = Header(stamp=node.get_clock().now().to_msg(), frame_id='link_base')
            for pub, points in zip(publishers, [white, yellow]):
                pub.publish(point_cloud2.create_cloud_xyz32(header, points))
            until = time.monotonic() + 0.1
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.02)
        spin_until(lambda: all(t in latest for t in topics) and
                   all(latest[t].poses for t in topics if t.startswith('/path/')))
        for color in ['white', 'yellow']:
            raw = latest[f'/lane_map/{color}']
            skel = latest[f'/lane_map/{color}_skeleton']
            assert raw.header.frame_id == skel.header.frame_id == 'odom'
            assert raw.info == skel.info
            assert 0 < skel.data.count(100) < raw.data.count(100)
        assert latest['/lane_map/white'].data != latest['/lane_map/yellow'].data
        for topic in topics[-2:]:
            assert latest[topic].header.frame_id == 'link_base'
            assert all(p.pose.position.x >= 0 for p in latest[topic].poses)

        # Stop map writers and explicitly send an empty synchronized map pair.
        for process in processes[:2]:
            process.terminate()
            process.wait(timeout=5)
        processes = processes[2:]
        clear_pubs = [node.create_publisher(OccupancyGrid, t, qos) for t in topics[2:4]]
        spin_until(lambda: all(p.get_subscription_count() for p in clear_pubs))
        empty = OccupancyGrid()
        empty.header = Header(stamp=node.get_clock().now().to_msg(), frame_id='odom')
        empty.info = latest['/lane_map/white'].info
        empty.data = [0] * (empty.info.width * empty.info.height)
        for pub in clear_pubs:
            pub.publish(empty)
        spin_until(lambda: all(latest[t].header.stamp == empty.header.stamp and not latest[t].poses
                               for t in topics[-2:]))
        print('ROS map pipeline passed: separate maps, thinning, rotated TF, both paths, empty-map clearing')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        for log in logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
