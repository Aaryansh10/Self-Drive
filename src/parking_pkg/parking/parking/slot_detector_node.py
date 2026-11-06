#!/usr/bin/env python3
"""ROS 2 node: /pc_whitelane (link_base) -> tracked parking slots in the odom frame.

Publishes
  /parking/slots       geometry_msgs/PoseArray   entrance centre, x-axis pointing INTO the slot
  /parking/slot_info   std_msgs/Float32MultiArray  per slot: [id, width, depth, confidence, back_seen, hits, locked]
  /parking/markers     visualization_msgs/MarkerArray  (RViz)
  /parking/map_points  sensor_msgs/PointCloud2   accumulated line map (debug)
Services
  /parking/lock_best   std_srvs/Trigger  freeze the most confident slot (stops smoothing)
  /parking/unlock      std_srvs/Trigger
"""
import math
from dataclasses import fields

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from parking.slot_core import (Params, SlotTracker, VoxelMap, extract_segments,
                               find_slots, parse_xy, split_at_junctions, to_odom)


def yaw_from_quat(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class SlotDetector(Node):
    def __init__(self):
        super().__init__("slot_detector")
        d = Params()
        for f in fields(Params):
            v = getattr(d, f.name)
            self.declare_parameter(f.name, list(v) if isinstance(v, tuple) else v)
        self.declare_parameter("cloud_topic", "/pc_whitelane")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("min_range", 0.8)       # in link_base, ignore points closer than this
        self.declare_parameter("max_range", 10.0)      # IPM gets noisy far away
        self.declare_parameter("detect_rate", 3.0)
        self.declare_parameter("map_radius", 30.0)
        self.declare_parameter("publish_min_conf", 0.15)
        self.p = self._read_params()
        self.world = self.get_parameter("world_frame").value
        self.map = VoxelMap(self.p.voxel)
        self.tracker = SlotTracker()
        self.robot_xy = None
        self.segs = []
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(PointCloud2, self.get_parameter("cloud_topic").value, self.on_cloud, qos)
        self.pub_slots = self.create_publisher(PoseArray, "/parking/slots", 5)
        self.pub_info = self.create_publisher(Float32MultiArray, "/parking/slot_info", 5)
        self.pub_mk = self.create_publisher(MarkerArray, "/parking/markers", 5)
        self.pub_map = self.create_publisher(PointCloud2, "/parking/map_points", 1)
        self.create_service(Trigger, "/parking/lock_best", self.on_lock)
        self.create_service(Trigger, "/parking/unlock", self.on_unlock)
        self.create_timer(1.0 / self.get_parameter("detect_rate").value, self.on_timer)
        self.get_logger().info("slot_detector ready")

    # -------------------------------------------------------------- params
    def _read_params(self):
        kw = {}
        for f in fields(Params):
            v = self.get_parameter(f.name).value
            kw[f.name] = tuple(v) if isinstance(v, (list, tuple)) else v
        return Params(**kw)

    # --------------------------------------------------------------- input
    def on_cloud(self, msg: PointCloud2):
        try:
            tf = self.tf_buf.lookup_transform(self.world, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                                              timeout=Duration(seconds=0.05))
        except Exception:
            try:
                tf = self.tf_buf.lookup_transform(self.world, msg.header.frame_id, Time())
            except Exception as e:  # noqa
                self.get_logger().warn(f"no TF {self.world}<-{msg.header.frame_id}: {e}", throttle_duration_sec=2.0)
                return
        n = msg.width * msg.height
        if n == 0:
            return
        offs = {f.name: f.offset for f in msg.fields}
        xy = parse_xy(bytes(msg.data), msg.point_step, n, (offs["x"], offs["y"]))
        r = np.hypot(xy[:, 0], xy[:, 1])
        xy = xy[(r > self.get_parameter("min_range").value) & (r < self.get_parameter("max_range").value)]
        t = tf.transform.translation
        yaw = yaw_from_quat(tf.transform.rotation)
        self.robot_xy = np.array([t.x, t.y])
        self.map.add(to_odom(xy, t.x, t.y, yaw))

    # ------------------------------------------------------------ detection
    def on_timer(self):
        if self.robot_xy is None:
            return
        self.p = self._read_params()
        self.map.prune(self.robot_xy, self.get_parameter("map_radius").value)
        pts = self.map.points(self.p.min_voxel_hits)
        if len(pts) < 20:
            return
        segs = extract_segments(pts, self.p)
        self.segs = split_at_junctions(segs, self.p)
        dets = find_slots(segs, self.p, ref_xy=self.robot_xy)
        tracks = self.tracker.update(dets)
        stamp = self.get_clock().now().to_msg()
        self.publish(tracks, pts, stamp)

    # ----------------------------------------------------------- publishing
    def publish(self, tracks, pts, stamp):
        minc = self.get_parameter("publish_min_conf").value
        good = [t for t in tracks if t.confidence >= minc]
        pa = PoseArray()
        pa.header.frame_id, pa.header.stamp = self.world, stamp
        info = Float32MultiArray()
        for t in good:
            s = t.slot
            pose = Pose()
            pose.position.x, pose.position.y = float(s.center[0]), float(s.center[1])
            pose.orientation.z, pose.orientation.w = math.sin(s.yaw / 2), math.cos(s.yaw / 2)
            pa.poses.append(pose)
            info.data += [float(t.id), float(s.width), float(s.depth), float(t.confidence),
                          float(s.back_seen), float(t.hits), float(t.locked)]
        self.pub_slots.publish(pa)
        self.pub_info.publish(info)
        self.pub_mk.publish(self.markers(good, stamp))
        pc = PointCloud2()
        pc.header.frame_id, pc.header.stamp = self.world, stamp
        pc.height, pc.width = 1, len(pts)
        pc.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                     for i, n in enumerate("xyz")]
        pc.point_step, pc.row_step, pc.is_dense = 12, 12 * len(pts), True
        pc.data = np.c_[pts, np.zeros(len(pts))].astype(np.float32).tobytes()
        self.pub_map.publish(pc)

    def markers(self, tracks, stamp):
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        def base(mid, mtype, ns):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.world, stamp
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        seg = base(0, Marker.LINE_LIST, "segments")
        seg.scale.x = 0.04
        seg.color.r = seg.color.g = seg.color.b = 0.7
        seg.color.a = 0.8
        for s in self.segs:
            for q in (s.p0, s.p1):
                from geometry_msgs.msg import Point
                seg.points.append(Point(x=float(q[0]), y=float(q[1]), z=0.02))
        ma.markers.append(seg)
        from geometry_msgs.msg import Point
        for t in tracks:
            s = t.slot
            poly = base(t.id * 10 + 1, Marker.LINE_STRIP, "slot")
            poly.scale.x = 0.08
            col = (0.2, 0.5, 1.0) if t.locked else ((0.1, 0.9, 0.2) if s.back_seen else (1.0, 0.8, 0.1))
            poly.color.r, poly.color.g, poly.color.b, poly.color.a = *col, 1.0
            for q in np.vstack([s.polygon(), s.polygon()[:1]]):
                poly.points.append(Point(x=float(q[0]), y=float(q[1]), z=0.05))
            ma.markers.append(poly)
            arr = base(t.id * 10 + 2, Marker.ARROW, "axis")
            arr.scale.x, arr.scale.y, arr.scale.z = 0.08, 0.18, 0.18
            arr.color.r, arr.color.g, arr.color.b, arr.color.a = *col, 1.0
            e = s.center + 1.2 * np.array([math.cos(s.yaw), math.sin(s.yaw)])
            arr.points = [Point(x=float(s.center[0]), y=float(s.center[1]), z=0.06),
                          Point(x=float(e[0]), y=float(e[1]), z=0.06)]
            ma.markers.append(arr)
            txt = base(t.id * 10 + 3, Marker.TEXT_VIEW_FACING, "label")
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(s.center[0]), float(s.center[1]), 0.8
            txt.scale.z = 0.3
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f"#{t.id} c={t.confidence:.2f} {'U' if s.back_seen else 'partial'} w={s.width:.2f} d={s.depth:.2f}"
            ma.markers.append(txt)
        return ma

    # -------------------------------------------------------------- services
    def on_lock(self, _req, resp):
        cand = [t for t in self.tracker.tracks if t.misses == 0]
        if not cand:
            resp.success, resp.message = False, "no visible slot track to lock"
            return resp
        best = max(cand, key=lambda t: t.confidence)
        self.tracker.lock(best.id)
        resp.success, resp.message = True, f"locked slot #{best.id}"
        return resp

    def on_unlock(self, _req, resp):
        self.tracker.lock(-1)
        resp.success, resp.message = True, "unlocked"
        return resp


def main():
    rclpy.init()
    node = SlotDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
