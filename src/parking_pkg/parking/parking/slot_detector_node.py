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
import threading
import time
from collections import deque
from dataclasses import fields

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
try:
    from map_msgs.msg import OccupancyGridUpdate
except ImportError:  # pragma: no cover
    OccupancyGridUpdate = None
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException, LookupException,
                     TransformListener)
from visualization_msgs.msg import Marker, MarkerArray

from parking.slot_core import (Params, SlotTracker, VoxelMap, extract_segments,
                               apply_grid_update, drain_pending, find_slots, grid_to_points, merge_duplicates,
                               parse_xy, split_at_junctions, to_odom)


def yaw_from_quat(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class SlotDetector(Node):
    def __init__(self):
        super().__init__("slot_detector")
        d = Params()
        for f in fields(Params):
            v = getattr(d, f.name)
            self.declare_parameter(f.name, list(v) if isinstance(v, tuple) else v)
        self.declare_parameter("input_mode", "map")        # "map" (team occupancy grid) or "cloud" (own accumulation)
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("robot_frame", "link_base")
        self.declare_parameter("map_window", 14.0)         # m, only this radius around the robot is processed
        self.declare_parameter("occ_threshold", 65)
        self.declare_parameter("cloud_topic", "/pc_whitelane")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("min_range", 0.8)       # in link_base, ignore points closer than this
        self.declare_parameter("max_range", 10.0)      # IPM gets noisy far away
        self.declare_parameter("detect_rate", 2.0)
        self.declare_parameter("map_radius", 30.0)
        self.declare_parameter("publish_min_conf", 0.4)
        self.declare_parameter("tf_max_wait", 0.5)         # s a cloud may wait for its TF before it is dropped
        self.declare_parameter("max_points", 5000)         # cap on points given to line fitting (bounds CPU)
        self.declare_parameter("max_yaw_rate", 0.3)        # rad/s, skip clouds while turning faster
        self.declare_parameter("nominal_depth", 0.0)       # >0: draw partial slots this deep (m)
        self.p = self._read_params()
        self.world = self.get_parameter("world_frame").value
        self.map = VoxelMap(self.p.voxel)
        self.tracker = SlotTracker()
        self.robot_xy = None
        self.prev = None
        self.dropped = 0
        self.ingested = 0
        self.det_ms = 0.0
        self.pending = deque()
        self.grid = None            # dict: data(2D int8), res, ox, oy, oyaw
        self.n_full = 0
        self.n_upd = 0
        self.n_upd_rejected = 0
        self.map_sub = None
        self.upd_sub = None
        self.cb_in = None
        self.mode = self.get_parameter("input_mode").value
        self.map_lock = threading.Lock()
        self.drain_lock = threading.Lock()
        self.rng = np.random.default_rng(0)
        self.segs = []
        self.tf_buf = Buffer()
        # own thread for TF so it is never starved by detection (this was the 2 s TF lag)
        self.tf_listener = TransformListener(self.tf_buf, self, spin_thread=True)
        cb_in = MutuallyExclusiveCallbackGroup()
        cb_det = MutuallyExclusiveCallbackGroup()
        self.cb_in = cb_in

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        if self.mode == "cloud":
            self.create_subscription(PointCloud2, self.get_parameter("cloud_topic").value, self.on_cloud, qos,
                                     callback_group=cb_in)
        # map mode: subscriptions are created lazily in on_timer, once the publisher exists,
        # with QoS copied from it (so a latched/transient-local map is received)
        self.pub_slots = self.create_publisher(PoseArray, "/parking/slots", 5)
        self.pub_info = self.create_publisher(Float32MultiArray, "/parking/slot_info", 5)
        self.pub_mk = self.create_publisher(MarkerArray, "/parking/markers", 5)
        self.pub_map = self.create_publisher(PointCloud2, "/parking/map_points", 1)
        self.create_service(Trigger, "/parking/lock_best", self.on_lock)
        self.create_service(Trigger, "/parking/unlock", self.on_unlock)
        self.create_timer(1.0 / self.get_parameter("detect_rate").value, self.on_timer, callback_group=cb_det)
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
        if msg.width * msg.height == 0:
            return
        self.pending.append((time.monotonic(), msg))
        self.drain()

    def qos_like(self, topic):
        infos = self.get_publishers_info_by_topic(topic)
        if not infos:
            return None
        q = infos[0].qos_profile
        latched = q.durability == DurabilityPolicy.TRANSIENT_LOCAL
        rel = ReliabilityPolicy.RELIABLE if (latched or q.reliability == ReliabilityPolicy.RELIABLE) \
            else ReliabilityPolicy.BEST_EFFORT
        return QoSProfile(reliability=rel, history=HistoryPolicy.KEEP_LAST, depth=5 if not latched else 1,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE)

    def try_subscribe_map(self):
        topic = self.get_parameter("map_topic").value
        if self.map_sub is None:
            q = self.qos_like(topic)
            if q is None:
                self.get_logger().warn(f"no publisher on {topic} yet", throttle_duration_sec=5.0)
                return
            self.map_sub = self.create_subscription(OccupancyGrid, topic, self.on_map, q, callback_group=self.cb_in)
            self.get_logger().info(f"subscribed {topic} (durability={q.durability.name}, reliability={q.reliability.name})")
        if self.upd_sub is None and OccupancyGridUpdate is not None:
            q = self.qos_like(topic + "_updates")
            if q is not None:
                self.upd_sub = self.create_subscription(OccupancyGridUpdate, topic + "_updates", self.on_update, q,
                                                        callback_group=self.cb_in)
                self.get_logger().info(f"subscribed {topic}_updates")

    def on_map(self, msg: OccupancyGrid):
        o = msg.info.origin
        arr = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        with self.map_lock:
            self.grid = dict(data=arr, res=msg.info.resolution, ox=o.position.x, oy=o.position.y,
                             oyaw=yaw_from_quat(o.orientation))
            self.world = msg.header.frame_id or self.world      # output in the map's own frame
            self.n_full += 1

    def on_update(self, msg):
        with self.map_lock:
            if self.grid is None:
                return          # patches are useless until a full map has been received
            if apply_grid_update(self.grid["data"], msg.x, msg.y, msg.width, msg.height, msg.data):
                self.n_upd += 1
            else:
                self.n_upd_rejected += 1

    def robot_in_world(self):
        try:    # latest available transform, non-blocking
            tf = self.tf_buf.lookup_transform(self.world, self.get_parameter("robot_frame").value, Time())
            return np.array([tf.transform.translation.x, tf.transform.translation.y])
        except Exception:  # noqa
            return None

    def lookup(self, msg):
        try:       # timeout=0 => never blocks
            tf = self.tf_buf.lookup_transform(self.world, msg.header.frame_id, Time.from_msg(msg.header.stamp))
            return "ok", tf
        except ExtrapolationException as e:
            return ("future" if "future" in str(e) else "fail"), None
        except (LookupException, ConnectivityException):
            return "fail", None

    def drain(self):
        if not self.drain_lock.acquire(blocking=False):
            return
        try:
            _, dropped = drain_pending(self.pending, self.lookup, self.ingest, time.monotonic(),
                                       max_wait=self.get_parameter("tf_max_wait").value)
            self.dropped += dropped
        finally:
            self.drain_lock.release()

    def ingest(self, msg, tf):
        stamp_t = Time.from_msg(msg.header.stamp)
        yaw_now = yaw_from_quat(tf.transform.rotation)
        tnow = stamp_t.nanoseconds * 1e-9
        t = tf.transform.translation
        self.robot_xy = np.array([t.x, t.y])
        if self.prev is not None:
            dt = tnow - self.prev[0]
            if dt > 1e-3:
                rate = abs(math.atan2(math.sin(yaw_now - self.prev[1]), math.cos(yaw_now - self.prev[1]))) / dt
                self.prev = (tnow, yaw_now)
                if rate > self.get_parameter("max_yaw_rate").value:
                    return          # fast rotation => IPM/odom registration is unreliable
        else:
            self.prev = (tnow, yaw_now)
        n = msg.width * msg.height
        offs = {f.name: f.offset for f in msg.fields}
        xy = parse_xy(bytes(msg.data), msg.point_step, n, (offs["x"], offs["y"]))
        r = np.hypot(xy[:, 0], xy[:, 1])
        xy = xy[(r > self.get_parameter("min_range").value) & (r < self.get_parameter("max_range").value)]
        with self.map_lock:
            self.map.add(to_odom(xy, t.x, t.y, yaw_now))
        self.ingested += 1

    # ------------------------------------------------------------ detection
    def on_timer(self):
        self.p = self._read_params()
        if self.mode == "map":
            self.try_subscribe_map()
            with self.map_lock:
                g = None if self.grid is None else dict(self.grid, data=self.grid["data"].copy())
            if g is None:
                self.get_logger().warn(
                    f"no full map received yet on {self.get_parameter('map_topic').value} "
                    f"(updates seen={self.n_upd}); if the map node only publishes the full grid rarely, "
                    "restart it or use input_mode: cloud", throttle_duration_sec=5.0)
                return
            self.robot_xy = self.robot_in_world()
            if self.robot_xy is None:
                self.get_logger().warn(f"no TF {self.world}<-{self.get_parameter('robot_frame').value}",
                                       throttle_duration_sec=5.0)
                return
            H, W = g["data"].shape
            pts = grid_to_points(g["data"].ravel(), W, H, g["res"], g["ox"], g["oy"], g["oyaw"], self.robot_xy,
                                 self.get_parameter("map_window").value, self.get_parameter("occ_threshold").value)
            self.get_logger().info(
                f"map {W}x{H}@{g['res']:.2f} frame={self.world} full={self.n_full} updates={self.n_upd} "
                f"rejected={self.n_upd_rejected} line_pts={len(pts)} last_detect={self.det_ms:.0f} ms",
                throttle_duration_sec=5.0)
        else:
            self.drain()                               # also flush clouds that were waiting for TF
            if self.robot_xy is None:
                return
            with self.map_lock:
                self.map.prune(self.robot_xy, self.get_parameter("map_radius").value)
                pts = self.map.points(self.p.min_voxel_hits)
        if self.mode == "cloud":
            self.get_logger().info(
                f"clouds ingested={self.ingested} dropped={self.dropped} pending={len(self.pending)} "
                f"map_pts={len(pts)} last_detect={self.det_ms:.0f} ms", throttle_duration_sec=5.0)
        if len(pts) < 20:
            return
        cap = int(self.get_parameter("max_points").value)
        if len(pts) > cap:
            pts = pts[self.rng.choice(len(pts), cap, replace=False)]
        t0 = time.monotonic()
        segs = extract_segments(pts, self.p)
        self.segs = split_at_junctions(merge_duplicates(segs, self.p), self.p)
        dets = find_slots(segs, self.p, ref_xy=self.robot_xy)
        tracks = self.tracker.update(dets)
        self.det_ms = (time.monotonic() - t0) * 1000.0
        self.publish(tracks, pts, self.get_clock().now().to_msg())

    # ----------------------------------------------------------- publishing
    def publish(self, tracks, pts, stamp):
        minc = self.get_parameter("publish_min_conf").value
        good = [t for t in tracks if t.confidence >= minc]
        pa = PoseArray()
        pa.header.frame_id, pa.header.stamp = self.world, stamp
        info = Float32MultiArray()
        vals = []
        for t in good:
            s = t.slot
            pose = Pose()
            pose.position.x, pose.position.y = float(s.center[0]), float(s.center[1])
            pose.orientation.z, pose.orientation.w = math.sin(s.yaw / 2), math.cos(s.yaw / 2)
            pa.poses.append(pose)
            vals += [float(t.id), float(s.width), float(s.depth), float(t.confidence),
                     float(s.back_seen), float(t.hits), float(t.locked)]
        info.data = vals
        if good:
            self.get_logger().info(
                "slots: " + ", ".join(f"#{t.id} c={t.confidence:.2f} {'U' if t.slot.back_seen else 'partial'}" for t in good),
                throttle_duration_sec=2.0)
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
            nd = self.get_parameter("nominal_depth").value
            P = s.polygon(max(s.depth, nd) if (nd > 0 and not s.back_seen) else None)
            for q in np.vstack([P, P[:1]]):
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
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
