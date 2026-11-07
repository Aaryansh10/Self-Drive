#!/usr/bin/env python3
"""Exercise real BT inputs/action client and path bridge against mock action servers.

Run from the workspace after building and sourcing install/setup.bash:
ROS_DOMAIN_ID=83 python3 src/bt/test/test_intersection_ros.py
"""
from pathlib import Path
import signal
import subprocess
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path as PathMsg
from nav2_msgs.action import FollowPath
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Bool, Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_broadcaster import TransformBroadcaster
from vision_interfaces.msg import Detection2D, DetectionArray
from intersection_interfaces_cpp.action import Intersection

ROOT = Path(__file__).resolve().parents[3]


def wait_for(predicate, timeout=6.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('Timed out waiting for integration condition')


def stop(process):
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    rclpy.init()
    node = rclpy.create_node('intersection_integration_test')
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    group = ReentrantCallbackGroup()
    processes = []
    logs = []
    extra_servers = []
    intersection_goals = []
    follow_goals = []
    canceled = []
    gate_values = []
    finish_intersection = threading.Event()
    cleanup = threading.Event()

    def execute_intersection(handle):
        intersection_goals.append(handle.request)
        while not finish_intersection.is_set() and not cleanup.is_set():
            if handle.is_cancel_requested:
                handle.canceled()
                return Intersection.Result(success=False)
            time.sleep(0.02)
        handle.succeed()
        return Intersection.Result(success=True)

    def execute_follow(handle):
        marker = handle.request.path.poses[-1].pose.position.x
        follow_goals.append(marker)
        while not cleanup.is_set():
            if handle.is_cancel_requested:
                canceled.append(marker)
                handle.canceled()
                return FollowPath.Result()
            time.sleep(0.02)
        handle.abort()
        return FollowPath.Result()

    intersection_server = ActionServer(node, Intersection, 'intersection', execute_intersection,
                                       callback_group=group, cancel_callback=lambda _: CancelResponse.ACCEPT)
    follow_server = ActionServer(node, FollowPath, 'follow_path', execute_follow,
                                 callback_group=group, cancel_callback=lambda _: CancelResponse.ACCEPT)
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    line_pub = node.create_publisher(PointCloud2,
                                     '/pc_stopline', qos_profile_sensor_data)
    sign_pub = node.create_publisher(DetectionArray, '/detections_depth', qos_profile_sensor_data)
    gate_sub = node.create_subscription(Bool, '/lane_follow/enable', lambda m: gate_values.append(m.data), latched)
    tf = StaticTransformBroadcaster(node)
    transform = TransformStamped()
    transform.header.frame_id = 'odom'
    transform.child_frame_id = 'link_base'
    transform.transform.rotation.w = 1.0
    tf.sendTransform(transform)

    def start(executable, args, name):
        log = open('/tmp/' + name + '.log', 'w+')
        logs.append(log)
        process = subprocess.Popen([str(ROOT / executable), '--ros-args', *args],
                                   stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def publish_inputs(probability=None, candidate=True, distance=3.0,
                       sign_distance=3.0, depth_valid=True, n_pts=11):
        header = Header(stamp=node.get_clock().now().to_msg(), frame_id='link_base')
        signs = DetectionArray(header=header)
        if probability is not None:
            d = Detection2D(class_name='sign', score=0.99, is_stop_sign=candidate,
                            sign_real_prob=probability)
            # Fusion has already filled this with stop-line depth for ALL signs.
            d.distance = sign_distance
            d.depth_valid = depth_valid
            d.n_pts = n_pts
            d.position.x = 100.0  # The sign image-ray position must not replace distance.
            signs.detections.append(d)
        sign_pub.publish(signs)
        time.sleep(0.04)
        header.stamp = node.get_clock().now().to_msg()
        if distance is not None:
            line_pub.publish(create_cloud_xyz32(header, [(distance, y / 10, 0.0) for y in range(-5, 6)]))

    bt_args = ['--params-file', str(ROOT / 'src/bt/config/bt.yaml'),
               '-p', 'tree_file:=' + str(ROOT / 'src/bt/trees/main.xml'), '-p', 'use_sim_time:=false']
    try:
        # Verify fusion's producer contract too: a fake sign with is_stop_sign=false
        # still receives stop-line depth, rather than physical-sign LiDAR depth.
        depth_messages = []
        depth_sub = node.create_subscription(DetectionArray, '/detections_depth',
                                             depth_messages.append, qos_profile_sensor_data)
        camera_pub = node.create_publisher(CameraInfo, '/zed_node/monocamera/camera_info', qos_profile_sensor_data)
        raw_pub = node.create_publisher(DetectionArray, '/detections', qos_profile_sensor_data)
        obstacle_pub = node.create_publisher(PointCloud2, '/obstacle_cloud', qos_profile_sensor_data)
        fusion = start('build/lidar_cam_fusion/fusion_node', [], 'intersection_fusion')
        wait_for(lambda: camera_pub.get_subscription_count() > 0 and raw_pub.get_subscription_count() > 0)
        for _ in range(20):
            header = Header(stamp=node.get_clock().now().to_msg(), frame_id='link_base')
            camera_pub.publish(CameraInfo(header=header, width=640, height=480,
                                          k=[100.0, 0.0, 320.0, 0.0, 100.0, 240.0, 0.0, 0.0, 1.0]))
            time.sleep(0.02)
            line_pub.publish(create_cloud_xyz32(header, [(2.5, y / 10, 0.0) for y in range(-5, 6)]))
            time.sleep(0.02)
            signs = DetectionArray(header=header)
            for genuine in [True, False]:
                signs.detections.append(Detection2D(class_name='sign', score=0.99,
                                                    is_stop_sign=genuine,
                                                    sign_real_prob=0.99 if genuine else 0.01))
            raw_pub.publish(signs)
            obstacle_pub.publish(create_cloud_xyz32(header, [(8.0, 0.0, 2.0)]))
            time.sleep(0.04)
            if depth_messages and all(d.depth_valid for d in depth_messages[-1].detections):
                break
        assert depth_messages, 'Fusion published no depth detections'
        assert len(depth_messages[-1].detections) == 2
        for d in depth_messages[-1].detections:
            assert d.depth_valid and abs(d.distance - 2.5) < 0.001 and d.n_pts == 11
        stop(fusion)
        processes.remove(fusion)
        node.destroy_subscription(depth_sub)
        print('PASS: fusion uses stop-line distance for real and fake sign detections')

        # Fake and unclassified signs use detection.distance without a stop action.
        for probability, candidate in [(0.01, True), (0.0, False), (None, False)]:
            finish_intersection.clear()
            before = len(intersection_goals)
            bt = start('build/bt/bt_executor', bt_args, 'intersection_bt_' + str(before))
            wait_for(lambda: line_pub.get_subscription_count() > 0 and sign_pub.get_subscription_count() > 0)
            time.sleep(0.4)
            if probability is not None:
                wait_for(lambda: gate_values and gate_values[-1])
                for _ in range(8):
                    publish_inputs(probability, candidate, distance=0.5, sign_distance=10.0)
                assert len(intersection_goals) == before, 'Cloud overrode the sign distance'
                assert gate_values[-1] is True
            for _ in range(8):
                # Sign detections must trigger without ANY cloud reaching the BT.
                publish_inputs(probability, candidate,
                               distance=0.5 if probability is None else None, sign_distance=0.5)
            wait_for(lambda: len(intersection_goals) == before + 1)
            assert intersection_goals[-1].turn_direction == Intersection.Goal.TURN_LEFT
            assert abs(intersection_goals[-1].road_width - 7.4) < 0.001
            finish_intersection.set()
            wait_for(lambda: gate_values and gate_values[-1])
            for _ in range(5):
                publish_inputs(probability, candidate, distance=0.5, sign_distance=0.5)
            assert len(intersection_goals) == before + 1, 'Repeated same stop line retriggered'
            stop(bt)
            processes.remove(bt)
            time.sleep(0.3)
        # A real sign must wait at the unimplemented stop placeholder, not bypass it.
        before = len(intersection_goals)
        bt = start('build/bt/bt_executor', bt_args, 'intersection_bt_real')
        wait_for(lambda: line_pub.get_subscription_count() > 0)
        time.sleep(0.4)
        wait_for(lambda: gate_values and gate_values[-1])
        # Valid but distant, invalid, non-finite, negative, and unsupported depths
        # must not be replaced by a nearby raw cloud or default zero distance.
        for depth, valid, count in [(10.0, True, 11), (0.5, False, 11),
                                    (float('nan'), True, 11), (-1.0, True, 11), (0.5, True, 0)]:
            for _ in range(4):
                publish_inputs(0.99, distance=0.5, sign_distance=depth, depth_valid=valid, n_pts=count)
            assert gate_values[-1] is True
            assert len(intersection_goals) == before
        for _ in range(10):
            publish_inputs(0.99, distance=None, sign_distance=2.75)
        time.sleep(0.4)
        assert len(intersection_goals) == before
        assert gate_values[-1] is False
        stop(bt)
        processes.remove(bt)
        real_log = Path('/tmp/intersection_bt_real.log').read_text()
        assert 'stopline at 2.75 m' in real_log, real_log
        print('PASS: real/fake sign detection distances, invalid-depth rejection, no-sign cloud fallback, and real-sign stop sequencing')

        # Reproduce startup history gaps: one TF sample newer than the detection.
        # A 31.25 ms gap is allowed for the memory anchor; 250 ms must be rejected.
        dynamic_tf = TransformBroadcaster(node)
        for index, skew in enumerate([0.03125, 0.25]):
            finish_intersection.clear()
            before = len(intersection_goals)
            name = 'intersection_tf_startup_' + str(index)
            bt = start('build/bt/bt_executor', bt_args + ['-p', 'confirm_intersection:=1'], name)
            wait_for(lambda: 'bt running:' in Path('/tmp/' + name + '.log').read_text())
            time.sleep(0.3)
            stamp = node.get_clock().now()
            sample = TransformStamped()
            sample.header.frame_id = 'odom'
            sample.child_frame_id = 'startup_camera_' + str(index)
            sample.header.stamp = (stamp + Duration(seconds=skew)).to_msg()
            sample.transform.rotation.w = 1.0
            dynamic_tf.sendTransform(sample)
            time.sleep(0.12)
            msg = DetectionArray(header=Header(stamp=stamp.to_msg(), frame_id=sample.child_frame_id))
            msg.detections.append(Detection2D(class_name='sign', score=1.0, is_stop_sign=False,
                                              sign_real_prob=0.0, depth_valid=True, n_pts=11, distance=2.25))
            sign_pub.publish(msg)
            if skew < 0.1:
                wait_for(lambda: len(intersection_goals) == before + 1)
                assert 'stopline at 2.25 m' in Path('/tmp/' + name + '.log').read_text()
                finish_intersection.set()
                wait_for(lambda: gate_values and gate_values[-1])
            else:
                time.sleep(0.3)
                assert len(intersection_goals) == before, 'Accepted a transform outside the tolerance'
                assert 'extrapolation' in Path('/tmp/' + name + '.log').read_text().lower()
            assert bt.poll() is None
            stop(bt)
            processes.remove(bt)
            time.sleep(0.3)
        print('PASS: bounded startup TF fallback preserves detection distance and rejects large skew')

        # The server may come up after the BT. Discovery must allow more than
        # the wrapper's former 500 ms without emitting a false startup error.
        finish_intersection.clear()
        before = len(intersection_goals)
        bt = start('build/bt/bt_executor', bt_args + ['-p', 'intersection_action:=intersection_delayed'],
                   'intersection_delayed_startup')
        time.sleep(2.0)
        delayed = ActionServer(node, Intersection, 'intersection_delayed', execute_intersection,
                               callback_group=group, cancel_callback=lambda _: CancelResponse.ACCEPT)
        extra_servers.append(delayed)
        wait_for(lambda: 'bt running:' in Path('/tmp/intersection_delayed_startup.log').read_text())
        assert "'intersection_delayed' is not reachable" not in Path('/tmp/intersection_delayed_startup.log').read_text()
        for _ in range(5):
            publish_inputs(0.0, candidate=False, distance=None, sign_distance=2.5)
        wait_for(lambda: len(intersection_goals) == before + 1)
        finish_intersection.set()
        wait_for(lambda: gate_values and gate_values[-1])
        stop(bt)
        processes.remove(bt)
        print('PASS: intersection action server discovered when it starts two seconds after the BT')

        # Real bridge must prioritize and cancel intersection paths, then resume lanes.
        gate_pub = node.create_publisher(Bool, '/lane_follow/enable', latched)
        lane_pub = node.create_publisher(PathMsg, '/path/selected_centerline', 10)
        turn_pub = node.create_publisher(PathMsg, '/intersection_path', latched)
        gate_pub.publish(Bool(data=True))  # Test late-subscriber gate durability.
        bridge = start('build/mapping_cpp/path_bridge', ['-p', 'lane_update_period:=0.1'], 'intersection_bridge')
        wait_for(lambda: lane_pub.get_subscription_count() > 0 and turn_pub.get_subscription_count() > 0)

        def path(marker):
            msg = PathMsg(header=Header(stamp=node.get_clock().now().to_msg(), frame_id='odom'))
            pose = PoseStamped(header=msg.header)
            pose.pose.position.x = marker
            pose.pose.orientation.w = 1.0
            msg.poses = [pose, pose]
            return msg

        lane_pub.publish(path(10.0))
        wait_for(lambda: 10.0 in follow_goals)
        turn_pub.publish(path(20.0))
        wait_for(lambda: 20.0 in follow_goals)
        wait_for(lambda: 10.0 in canceled)
        gate_pub.publish(Bool(data=False))
        lane_pub.publish(path(30.0))
        time.sleep(0.3)
        assert 30.0 not in follow_goals and 20.0 not in canceled
        turn_pub.publish(PathMsg())
        wait_for(lambda: 20.0 in canceled)
        time.sleep(0.15)
        assert 30.0 not in follow_goals
        lane_pub.publish(path(30.0))
        gate_pub.publish(Bool(data=True))
        wait_for(lambda: 30.0 in follow_goals)
        stop(bridge)
        processes.remove(bridge)
        print('PASS: intersection path priority, lane gate ordering, cancellation, and lane resumption')
    finally:
        cleanup.set()
        finish_intersection.set()
        for process in processes:
            stop(process)
        executor.shutdown(timeout_sec=3)
        thread.join(timeout=3)
        for server in extra_servers:
            server.destroy()
        intersection_server.destroy()
        follow_server.destroy()
        node.destroy_subscription(gate_sub)
        node.destroy_node()
        rclpy.shutdown()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
