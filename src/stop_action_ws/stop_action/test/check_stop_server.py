#!/usr/bin/env python3
"""Live ROS regression check. Run with a built workspace sourced; uses ROS_DOMAIN_ID=87.
Starts its own stop server; no simulator or moving robot is needed.
"""
import os
os.environ['ROS_DOMAIN_ID'] = '87'
from pathlib import Path
import subprocess
import time
import rclpy
from ament_index_python.packages import get_package_prefix
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from vision_interfaces.msg import DetectionArray, Detection2D
from stop_interfaces.action import Stop


def main():
    rclpy.init()
    node = rclpy.create_node('stop_regression')
    odom = node.create_publisher(Odometry, '/odom', 10)
    lane = node.create_publisher(DetectionArray, '/detections_lane', 10)
    drive = node.create_publisher(Twist, '/cmd_vel_drive', 10)
    commands = []
    subscription = node.create_subscription(Twist, '/cmd_vel', lambda m: commands.append(m.linear.x), 10)
    client = ActionClient(node, Stop, 'stop')
    executable = Path(get_package_prefix('stop_action')) / 'lib/stop_action/stop_action'
    process = subprocess.Popen([str(executable), '--ros-args',
                                '--params-file', str(Path(__file__).resolve().parents[1] / 'config/stop_server.yaml'),
                                '-p', 'use_sim_time:=false', '-p', 'goal_timeout:=0.8', '-p', 'pedestrian_clear_time:=0.2',
                                '-p', 'detection_timeout:=0.3'])
    occupied = True
    send_lane = True

    def pump(duration):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            odom.publish(Odometry())
            msg = DetectionArray()
            if occupied:
                d = Detection2D()
                d.class_name = 'pedestrian'
                d.depth_valid = True
                d.is_in_lane = True
                msg.detections.append(d)
            if send_lane:
                lane.publish(msg)
            cmd = Twist()
            cmd.linear.x = 0.5
            drive.publish(cmd)
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.01)

    def goal(cls):
        g = Stop.Goal()
        g.target_class = cls
        g.obstacle_distance = 0.01
        g.hold_time = 9.0  # must not control pedestrian release or sign dwell
        future = client.send_goal_async(g)
        deadline = time.monotonic() + 3
        while not future.done() and time.monotonic() < deadline:
            pump(0.05)
        assert future.done(), 'goal response timed out'
        return future.result()

    try:
        assert client.wait_for_server(timeout_sec=10), 'server unavailable'
        pump(0.4)
        assert 0.5 in commands, 'navigation passthrough missing'
        assert not goal('barrel').accepted
        assert not goal('pothole').accepted
        assert not goal('stop_line').accepted  # line geometry is part of a stop_sign goal
        gh = goal('pedestrian')
        assert gh.accepted
        result = gh.get_result_async()
        pump(1.2)  # longer than approach timeout
        assert not result.done(), 'pedestrian released while occupied'
        commands.clear()
        pump(0.2)
        assert commands and all(v == 0 for v in commands), 'navigation overrides stop'
        send_lane = False
        occupied = False
        pump(0.6)
        assert not result.done(), 'stale lane data released pedestrian'
        send_lane = True
        pump(0.5)
        assert result.done() and result.result().result.success
        gh = goal('stop_sign')
        assert gh.accepted
        result = gh.get_result_async()
        pump(1.6)
        assert not result.done(), 'timed stop ended before two seconds'
        pump(0.7)
        assert result.done() and result.result().result.success, 'timed stop did not finish'
        occupied = True
        gh = goal('pedestrian')
        pump(0.3)
        canceled = gh.cancel_goal_async()
        pump(0.3)
        assert canceled.done() and canceled.result().goals_canceling
        print('PASS: passthrough, override, class rejection, pedestrian occupancy/staleness, two-second dwell, cancellation')
    finally:
        process.terminate()
        process.wait(timeout=5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
