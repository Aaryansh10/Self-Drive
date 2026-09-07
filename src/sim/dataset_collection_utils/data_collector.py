#!/usr/bin/env python3
"""
Subscribes to a camera image topic, shows a live preview window, and saves
the current frame to disk every time you press 'p'.

Filenames are sequential: img_000000.png, img_000001.png, ...
On startup it scans the output directory for existing img_XXXXXX.png files
and continues numbering from the highest one found + 1, so you can stop
and restart the script across multiple sessions without overwriting or
restarting from zero.

Usage:
    python3 collect_data.py --topic /camera/image_raw --outdir ./dataset

Controls (with the preview window focused):
    p   -> save current frame
    q   -> quit
"""

import argparse
import os
import re
import threading

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

FILENAME_PATTERN = re.compile(r"^img_(\d{6})\.png$")


def get_next_index(outdir: str) -> int:
    """Look at existing img_XXXXXX.png files and return the next free index."""
    if not os.path.isdir(outdir):
        return 0
    highest = -1
    for fname in os.listdir(outdir):
        match = FILENAME_PATTERN.match(fname)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


class DataCollector(Node):
    def __init__(self, topic: str, outdir: str):
        super().__init__("data_collector")
        self.outdir = outdir
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        os.makedirs(self.outdir, exist_ok=True)
        self.next_index = get_next_index(self.outdir)
        self.get_logger().info(
            f"Resuming from index {self.next_index} in '{self.outdir}'"
        )

        self.subscription = self.create_subscription(
            Image, topic, self.image_callback, 10
        )
        self.get_logger().info(f"Subscribed to '{topic}'")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return
        with self.frame_lock:
            self.latest_frame = frame

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def save_frame(self, frame):
        filename = f"img_{self.next_index:06d}.png"
        filepath = os.path.join(self.outdir, filename)
        cv2.imwrite(filepath, frame)
        self.get_logger().info(f"Saved {filepath}")
        self.next_index += 1


def main():
    parser = argparse.ArgumentParser(description="Collect training images from a camera topic.")
    parser.add_argument(
        "--topic", default="/stereocamera/image_raw",
        help="Image topic to subscribe to (default: /camera/image_raw)",
    )
    parser.add_argument(
        "--outdir", default="./dataset",
        help="Directory to save images into (default: ./dataset)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = DataCollector(args.topic, args.outdir)

    # Spin ROS in a background thread so the main thread is free to run
    # the OpenCV window + keypress loop.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    window_name = "Camera Feed - press 'p' to save, 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            frame = node.get_latest_frame()
            if frame is not None:
                cv2.imshow(window_name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('p'):
                if frame is not None:
                    node.save_frame(frame)
                else:
                    node.get_logger().warn("No frame received yet, nothing to save.")
            elif key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()