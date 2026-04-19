#!/usr/bin/env python3
"""
aruco_overlay.py
Subscribes to:
  - /image_raw/compressed      (sensor_msgs/CompressedImage)
  - /aruco_markers             (ros2_aruco_interfaces/ArucoMarkers)

Displays a live OpenCV window with bounding boxes and marker IDs
drawn over the camera feed.

Usage:
  python3 aruco_overlay.py --camera cam_right

Dependencies:
  pip install opencv-contrib-python
  (ros2_aruco and its interfaces must already be built in your workspace)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
import threading

from sensor_msgs.msg import CompressedImage, CameraInfo
from ros2_aruco_interfaces.msg import ArucoMarkers
from cv_bridge import CvBridge
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=str, choices=['cam_left', 'cam_right', 'rpi'], required=True, help='Camera to use')
args = parser.parse_args()

if args.camera == 'cam_left':
    image_topic        = '/cam_left/image_raw/compressed'
    markers_topic      = '/cam_left/aruco_markers'
    camera_info_topic  = '/cam_left/camera_info'
elif args.camera == 'cam_right':
    image_topic        = '/cam_right/image_raw/compressed'
    markers_topic      = '/cam_right/aruco_markers'
    camera_info_topic  = '/cam_right/camera_info'
elif args.camera == 'rpi':
    image_topic        = '/rpi/image_raw/compressed'
    markers_topic      = '/rpi/aruco_markers'
    camera_info_topic  = '/rpi/camera_info'
else:
    sys.exit(0)

# ── visual config ─────────────────────────────────────────────────────────────
BOX_COLOR      = (0, 255, 120)
BOX_THICKNESS  = 2
LABEL_COLOR    = (0, 255, 120)
LABEL_BG       = (0, 0, 0)
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.65
FONT_THICKNESS = 2
CORNER_LEN     = 18
WINDOW_TITLE   = args.camera + ' visualisation'
# ─────────────────────────────────────────────────────────────────────────────


class ArucoOverlayNode(Node):
    def __init__(self):
        super().__init__("aruco_overlay")

        self.bridge      = CvBridge()
        self.lock        = threading.Lock()
        self.frame       = None
        self.markers     = []
        self.marker_size = 0.0625

        self.fx = self.fy = None
        self.cx = self.cy = None
        self.dist_coeffs  = np.zeros((4, 1))

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            CompressedImage,
            image_topic,
            self._image_cb,
            best_effort_qos,
        )

        self.create_subscription(
            ArucoMarkers,
            markers_topic,
            self._markers_cb,
            10,
        )

        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._camera_info_cb,
            10,
        )

        self.get_logger().info(f"aruco_overlay started on {image_topic}")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("imdecode returned None")
        except Exception as e:
            self.get_logger().error(f"compressed decode error: {e}")
            return
        with self.lock:
            self.frame = frame.copy()

    def _markers_cb(self, msg: ArucoMarkers):
        with self.lock:
            self.markers = list(zip(msg.marker_ids, msg.poses))
        self.get_logger().info(f"Received {len(msg.marker_ids)} markers")

    def _camera_info_cb(self, msg: CameraInfo):
        with self.lock:
            if self.fx is not None:
                return
            k = msg.k
            self.fx, self.fy = k[0], k[4]
            self.cx, self.cy = k[2], k[5]
            d = msg.d
            self.dist_coeffs = np.array(d[:4], dtype=np.float64).reshape(4, 1) if d else np.zeros((4, 1))
            self.get_logger().info(f"Got camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f}")

    # ── rendering ─────────────────────────────────────────────────────────────

    def get_annotated_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            frame   = self.frame.copy()
            markers = list(self.markers)
            fx, fy  = self.fx, self.fy
            cx, cy  = self.cx, self.cy
            dist    = self.dist_coeffs.copy()
            msize   = self.marker_size

        h, w = frame.shape[:2]

        if fx is None:
            fx = fy = w * 1.2
            cx, cy = w / 2.0, h / 2.0
            dist = np.zeros((4, 1))

        camera_matrix = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0,  0,  1]], dtype=np.float64)

        half = msize / 2.0
        obj_corners = np.array([[-half,  half, 0],
                                 [ half,  half, 0],
                                 [ half, -half, 0],
                                 [-half, -half, 0]], dtype=np.float64)
        z_tip = np.array([[0, 0, msize]], dtype=np.float64)

        for marker_id, pose in markers:
            # ── Convert ROS pose back to OpenCV rvec/tvec ─────────────────
            # Your aruco_node remapped: ROS_X=OpenCV_Z, ROS_Y=-OpenCV_X, ROS_Z=-OpenCV_Y
            # So to reverse: OpenCV_X=-ROS_Y, OpenCV_Y=-ROS_Z, OpenCV_Z=ROS_X
            ros_x = pose.position.x
            ros_y = pose.position.y
            ros_z = pose.position.z

            tvec = np.array([
                [-ros_y],   # OpenCV X = -ROS Y
                [-ros_z],   # OpenCV Y = -ROS Z
                [ ros_x],   # OpenCV Z =  ROS X
            ], dtype=np.float64)

            # Reverse the axis_remap rotation:
            # axis_remap was: [[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]]
            # inverse (transpose) is: [[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]]
            axis_remap_inv = np.array([
                [ 0, -1,  0,  0],
                [ 0,  0, -1,  0],
                [ 1,  0,  0,  0],
                [ 0,  0,  0,  1]
            ], dtype=np.float64)

            q = pose.orientation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            rot_matrix = np.eye(4)
            rot_matrix[0:3, 0:3] = np.array([
                [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
                [  2*(qx*qy + qz*qw),   1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
                [  2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy)],
            ], dtype=np.float64)

            # Undo the remap applied in aruco_node
            rot_matrix = axis_remap_inv @ rot_matrix
            rvec, _ = cv2.Rodrigues(rot_matrix[0:3, 0:3])

            # ── Project corners ────────────────────────────────────────────
            img_corners, _ = cv2.projectPoints(
                obj_corners, rvec, tvec, camera_matrix, dist)
            img_corners = img_corners.reshape(-1, 2).astype(int)

            # Bounding box
            rect = cv2.boundingRect(img_corners)
            x1, y1, bw, bh = rect
            x2, y2 = x1 + bw, y1 + bh
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            _draw_corner_ticks(frame, x1, y1, x2, y2, BOX_COLOR, CORNER_LEN)

            for pt in img_corners:
                cv2.circle(frame, tuple(pt), 4, BOX_COLOR, -1)

            cx_m = int(img_corners[:, 0].mean())
            cy_m = int(img_corners[:, 1].mean())
            cv2.drawMarker(frame, (cx_m, cy_m), BOX_COLOR,
                           cv2.MARKER_CROSS, 12, 1)

            # ── Z-axis arrow ───────────────────────────────────────────────
            origin_2d = (cx_m, cy_m)
            z_pt, _ = cv2.projectPoints(
                z_tip, rvec, tvec, camera_matrix, dist)
            z_pt = tuple(z_pt.reshape(2).astype(int))
            cv2.arrowedLine(frame, origin_2d, z_pt,
                            color=(255, 80, 0),
                            thickness=2,
                            tipLength=0.3)

            # Distance label — use ROS X which is forward distance
            dist_label = f"dist: {ros_x:.3f}m"
            (tw, th), _ = cv2.getTextSize(dist_label, FONT, 0.55, 1)
            lx = z_pt[0] - tw // 2
            ly = z_pt[1] - 8
            cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2),
                          (0, 0, 0), -1)
            cv2.putText(frame, dist_label, (lx, ly),
                        FONT, 0.55, (255, 80, 0), 1, cv2.LINE_AA)

            # ── ID label ───────────────────────────────────────────────────
            label = f"ID: {marker_id}"
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            lx2 = x1
            ly2 = max(y1 - 6, th + 4)
            cv2.rectangle(frame,
                          (lx2, ly2 - th - 4), (lx2 + tw + 6, ly2 + 2),
                          LABEL_BG, -1)
            cv2.putText(frame, label, (lx2 + 3, ly2 - 2),
                        FONT, FONT_SCALE, LABEL_COLOR, FONT_THICKNESS,
                        cv2.LINE_AA)

        # HUD
        hud = f"Markers detected: {len(markers)}"
        cv2.putText(frame, hud, (10, h - 12),
                    FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        return frame


# ── helpers ───────────────────────────────────────────────────────────────────

def _draw_corner_ticks(img, x1, y1, x2, y2, color, length):
    t = BOX_THICKNESS
    corners = [(x1, y1, 1, 1), (x2, y1, -1, 1),
               (x1, y2, 1, -1), (x2, y2, -1, -1)]
    for (cx, cy, dx, dy) in corners:
        cv2.line(img, (cx, cy), (cx + dx * length, cy),               color, t)
        cv2.line(img, (cx, cy), (cx,               cy + dy * length), color, t)


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ArucoOverlayNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 540)

    try:
        while rclpy.ok():
            frame = node.get_annotated_frame()

            if frame is None:
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder,
                            "Waiting for camera ...",
                            (60, 180), FONT, 0.8, (120, 120, 120), 2)
                cv2.imshow(WINDOW_TITLE, placeholder)
            else:
                cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()