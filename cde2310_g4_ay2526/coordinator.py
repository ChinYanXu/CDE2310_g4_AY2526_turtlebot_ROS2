#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose

from ros2_aruco_interfaces.msg import ArucoMarkers

from cde2310_g4_ay2526.frontier_detection import (
    OccupancyGrid2d,
    detect_frontiers,
    choose_frontier,
    choose_fallback_viewpoint,
)


@dataclass
class DetectionRecord:
    frame_id: str
    stamp_sec: float
    marker_id: int
    pose_stamped: PoseStamped
    source: str


class Coordinator(Node):
    def __init__(self):
        super().__init__('coordinator_main')

        # frontier detection params
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('planner_period_sec', 0.5)
        self.declare_parameter('min_frontier_size', 3)
        self.declare_parameter('frontier_strategy', 'largest')
        self.declare_parameter('fallback_min_obstacle_clearance_cells', 2)
        self.declare_parameter('fallback_revisit_radius_m', 0.8)
        self.declare_parameter('max_recent_fallbacks', 15)

        # aruco tag and offset params
        self.declare_parameter('stationary_tag_id', 1)
        self.declare_parameter('moving_tag_id', 2)
        self.declare_parameter('midpoint_tag_id', 3)

        self.declare_parameter('goal_standoff_m', 0.28)
        self.declare_parameter('midpoint_standoff_m', 0.30)

        self.declare_parameter('detection_timeout_sec', 0.8)

        # Camera-frame release gate.
        # Adjust axes/thresholds after testing.
        self.declare_parameter('release_lateral_threshold_m', 0.05)
        self.declare_parameter('release_forward_min_m', 0.15)
        self.declare_parameter('release_forward_max_m', 0.32)
        self.declare_parameter('release_required_consecutive_frames', 3)

        # Which camera topic should be trusted for release checks.
        self.declare_parameter('release_camera_source', 'rpi')

        # Optional topic to trigger dispenser.
        self.declare_parameter('dispense_topic', '/dispense_cmd')

        map_topic = self.get_parameter('map_topic').value
        self.dispense_topic = self.get_parameter('dispense_topic').value

        # ===== Subscriptions =====
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, 10
        )

        self.aruco_left_sub = self.create_subscription(
            ArucoMarkers, '/cam_left/aruco_markers', self.aruco_left_cb, 10
        )
        self.aruco_right_sub = self.create_subscription(
            ArucoMarkers, '/cam_right/aruco_markers', self.aruco_right_cb, 10
        )
        self.aruco_rpi_sub = self.create_subscription(
            ArucoMarkers, '/rpi/aruco_markers', self.aruco_rpi_cb, 10
        )

        # ===== Publishers / actions =====
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        from std_msgs.msg import Bool
        self.dispense_pub = self.create_publisher(Bool, self.dispense_topic, 10)

        # ===== TF =====
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ===== Timer =====
        self.timer = self.create_timer(
            float(self.get_parameter('planner_period_sec').value),
            self.control_loop
        )

        # ===== State =====
        self.map_msg: Optional[OccupancyGrid] = None
        self.goal_handle = None
        self.nav_busy = False
        self.last_goal_type: Optional[str] = None
        self.recent_fallback_points = []

        self.state = 'EXPLORE'
        # EXPLORE, GO_TO_STATIONARY, GO_TO_MIDPOINT, WAIT_FOR_MOVING_TIN

        self.latest_detections: Dict[int, DetectionRecord] = {}
        self.release_frame_counter = 0
        self.last_dispense_time_sec = -999.0

        self.get_logger().info('Coordinator node started.')

    # ------------------------------------------------------------------
    # Basic callbacks
    # ------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg

    def aruco_left_cb(self, msg: ArucoMarkers):
        self.handle_aruco_msg(msg, 'cam_left')

    def aruco_right_cb(self, msg: ArucoMarkers):
        self.handle_aruco_msg(msg, 'cam_right')

    def aruco_rpi_cb(self, msg: ArucoMarkers):
        self.handle_aruco_msg(msg, 'rpi')

    def handle_aruco_msg(self, msg: ArucoMarkers, source: str):
        now_sec = self.now_sec()

        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = pose

            new_record = DetectionRecord(
                frame_id=msg.header.frame_id,
                stamp_sec=now_sec,
                marker_id=int(marker_id),
                pose_stamped=ps,
                source=source,
            )

            old = self.latest_detections.get(int(marker_id))
            if old is None:
                self.latest_detections[int(marker_id)] = new_record
                continue

            # Prefer fresher detections. If same age, prefer release camera.
            release_source = self.get_parameter('release_camera_source').value
            if new_record.stamp_sec > old.stamp_sec:
                self.latest_detections[int(marker_id)] = new_record
            elif source == release_source and old.source != release_source:
                self.latest_detections[int(marker_id)] = new_record

    # ------------------------------------------------------------------
    # Time / TF helpers
    # ------------------------------------------------------------------

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def get_robot_pose_in_map(self) -> Optional[PoseStamped]:
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.3)
            )
        except TransformException as ex:
            self.get_logger().warn(f'Could not get map->base_link transform: {ex}')
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.position.z = t.transform.translation.z
        pose.pose.orientation = t.transform.rotation
        return pose

    def get_fresh_detection(self, marker_id: int) -> Optional[DetectionRecord]:
        rec = self.latest_detections.get(marker_id)
        if rec is None:
            return None

        timeout_sec = float(self.get_parameter('detection_timeout_sec').value)
        if self.now_sec() - rec.stamp_sec > timeout_sec:
            return None

        return rec

    def get_tag_pose_in_map(self, marker_id: int) -> Optional[PoseStamped]:
        rec = self.get_fresh_detection(marker_id)
        if rec is None:
            return None

        try:
            tf = self.tf_buffer.lookup_transform(
                'map',
                rec.pose_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )
            transformed = do_transform_pose(rec.pose_stamped, tf)
            transformed.header.frame_id = 'map'
            return transformed
        except Exception as ex:
            self.get_logger().warn(
                f'Could not transform tag {marker_id} from '
                f'{rec.pose_stamped.header.frame_id} to map: {ex}'
            )
            return None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def make_offset_goal(
        self,
        target_pose_map: PoseStamped,
        stand_off: float
    ) -> Optional[PoseStamped]:
        robot_pose = self.get_robot_pose_in_map()
        if robot_pose is None:
            return None

        tx = target_pose_map.pose.position.x
        ty = target_pose_map.pose.position.y
        rx = robot_pose.pose.position.x
        ry = robot_pose.pose.position.y

        yaw = math.atan2(ty - ry, tx - rx)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = tx - stand_off * math.cos(yaw)
        goal.pose.position.y = ty - stand_off * math.sin(yaw)
        goal.pose.position.z = 0.0
        goal.pose.orientation = self.yaw_to_quaternion(yaw)
        return goal

    # ------------------------------------------------------------------
    # Nav2 goal helpers
    # ------------------------------------------------------------------

    def send_nav_goal(self, goal_pose: PoseStamped):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('NavigateToPose action server not available yet.')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.nav_busy = True
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('Goal rejected.')
            self.nav_busy = False
            self.goal_handle = None
            return

        self.goal_handle = goal_handle
        self.get_logger().info('Goal accepted.')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        self.nav_busy = False

        result = future.result()
        if result is None:
            self.get_logger().warn('No result received from NavigateToPose.')
            self.goal_handle = None
            return

        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal reached successfully.')

            if self.last_goal_type == 'fallback' and self.goal_handle is not None:
                goal_pose = self.goal_handle.request.pose.pose.position
                self.recent_fallback_points.append((goal_pose.x, goal_pose.y))

                max_recent = int(self.get_parameter('max_recent_fallbacks').value)
                if len(self.recent_fallback_points) > max_recent:
                    self.recent_fallback_points.pop(0)

            if self.last_goal_type == 'midpoint':
                self.state = 'WAIT_FOR_MOVING_TIN'
                self.release_frame_counter = 0
                self.get_logger().info('Reached midpoint. Waiting for moving tin.')

        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Goal canceled.')
        else:
            self.get_logger().warn(f'Goal ended with status {status}.')
            if self.state != 'WAIT_FOR_MOVING_TIN':
                self.state = 'EXPLORE'

        self.goal_handle = None

    def cancel_current_goal(self):
        if self.goal_handle is None:
            return
        if not self.nav_busy:
            return

        self.get_logger().info('Canceling current goal...')
        cancel_future = self.goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        self.get_logger().info('Cancel request sent.')

    # ------------------------------------------------------------------
    # Dispense helpers
    # ------------------------------------------------------------------

    def trigger_dispense(self):
        # simple Bool pulse
        from std_msgs.msg import Bool
        msg = Bool()
        msg.data = True
        self.dispense_pub.publish(msg)
        self.last_dispense_time_sec = self.now_sec()
        self.release_frame_counter = 0
        self.get_logger().info('DISPENSE TRIGGERED')

    def pose_is_in_release_zone(self, marker_id: int) -> bool:
        rec = self.get_fresh_detection(marker_id)
        if rec is None:
            return False

        release_source = self.get_parameter('release_camera_source').value
        if rec.source != release_source:
            return False

        # ros2_aruco publishes tvec directly as pose.position.
        # In the common optical-camera convention used by this node:
        # x = horizontal, z = forward range for practical gating here.
        x = rec.pose_stamped.pose.position.x
        z = rec.pose_stamped.pose.position.z

        lat_thresh = float(self.get_parameter('release_lateral_threshold_m').value)
        z_min = float(self.get_parameter('release_forward_min_m').value)
        z_max = float(self.get_parameter('release_forward_max_m').value)

        return abs(x) <= lat_thresh and z_min <= z <= z_max

    def update_release_counter(self, marker_id: int) -> bool:
        needed = int(self.get_parameter('release_required_consecutive_frames').value)

        if self.pose_is_in_release_zone(marker_id):
            self.release_frame_counter += 1
        else:
            self.release_frame_counter = 0

        return self.release_frame_counter >= needed

    # ------------------------------------------------------------------
    # Frontier mode
    # ------------------------------------------------------------------

    def run_frontier_mode(self):
        if self.nav_busy:
            return

        if self.map_msg is None:
            self.get_logger().info('Waiting for map...')
            return

        robot_pose_stamped = self.get_robot_pose_in_map()
        if robot_pose_stamped is None:
            return

        costmap = OccupancyGrid2d(self.map_msg)

        frontiers = detect_frontiers(
            costmap,
            robot_pose_stamped.pose,
            min_frontier_size=int(self.get_parameter('min_frontier_size').value)
        )

        strategy = self.get_parameter('frontier_strategy').value
        chosen_frontier = choose_frontier(
            frontiers,
            robot_pose_stamped.pose,
            strategy=strategy
        )

        if chosen_frontier is not None:
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = chosen_frontier.x
            goal.pose.position.y = chosen_frontier.y
            goal.pose.position.z = 0.0

            dx = chosen_frontier.x - robot_pose_stamped.pose.position.x
            dy = chosen_frontier.y - robot_pose_stamped.pose.position.y
            yaw = math.atan2(dy, dx)
            goal.pose.orientation = self.yaw_to_quaternion(yaw)

            self.get_logger().info(
                f'[Frontier] Sending goal '
                f'x={goal.pose.position.x:.2f}, y={goal.pose.position.y:.2f}, '
                f'size={chosen_frontier.size}'
            )

            self.last_goal_type = 'frontier'
            self.send_nav_goal(goal)
            return

        fallback = choose_fallback_viewpoint(
            costmap=costmap,
            robot_pose=robot_pose_stamped.pose,
            recent_points=self.recent_fallback_points,
            min_clearance_cells=int(
                self.get_parameter('fallback_min_obstacle_clearance_cells').value
            ),
            revisit_radius=float(
                self.get_parameter('fallback_revisit_radius_m').value
            ),
        )

        if fallback is None:
            self.get_logger().warn(
                'No fallback viewpoint found. Exploration may be complete or trapped.'
            )
            return

        fx, fy = fallback

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = fx
        goal.pose.position.y = fy
        goal.pose.position.z = 0.0

        dx = fx - robot_pose_stamped.pose.position.x
        dy = fy - robot_pose_stamped.pose.position.y
        yaw = math.atan2(dy, dx)
        goal.pose.orientation = self.yaw_to_quaternion(yaw)

        self.get_logger().info(
            f'[Fallback] Sending goal x={fx:.2f}, y={fy:.2f}'
        )

        self.last_goal_type = 'fallback'
        self.send_nav_goal(goal)

    # ------------------------------------------------------------------
    # Main coordinator state machine
    # ------------------------------------------------------------------

    def control_loop(self):
        if self.map_msg is None:
            return

        stationary_tag_id = int(self.get_parameter('stationary_tag_id').value)
        moving_tag_id = int(self.get_parameter('moving_tag_id').value)
        midpoint_tag_id = int(self.get_parameter('midpoint_tag_id').value)

        goal_standoff_m = float(self.get_parameter('goal_standoff_m').value)
        midpoint_standoff_m = float(self.get_parameter('midpoint_standoff_m').value)

        # Priority:
        # 1) stationary tin if visible
        # 2) midpoint for moving tin
        # 3) exploration

        if self.state == 'EXPLORE':
            stationary_pose_map = self.get_tag_pose_in_map(stationary_tag_id)
            if stationary_pose_map is not None:
                if self.nav_busy:
                    self.cancel_current_goal()

                goal = self.make_offset_goal(stationary_pose_map, goal_standoff_m)
                if goal is not None:
                    self.get_logger().info(
                        f'[Stationary tag] Sending approach goal '
                        f'x={goal.pose.position.x:.2f}, y={goal.pose.position.y:.2f}'
                    )
                    self.last_goal_type = 'stationary'
                    if self.send_nav_goal(goal):
                        self.state = 'GO_TO_STATIONARY'
                        self.release_frame_counter = 0
                        return

            midpoint_pose_map = self.get_tag_pose_in_map(midpoint_tag_id)
            if midpoint_pose_map is not None:
                if self.nav_busy:
                    self.cancel_current_goal()

                goal = self.make_offset_goal(midpoint_pose_map, midpoint_standoff_m)
                if goal is not None:
                    self.get_logger().info(
                        f'[Midpoint tag] Sending approach goal '
                        f'x={goal.pose.position.x:.2f}, y={goal.pose.position.y:.2f}'
                    )
                    self.last_goal_type = 'midpoint'
                    if self.send_nav_goal(goal):
                        self.state = 'GO_TO_MIDPOINT'
                        self.release_frame_counter = 0
                        return

            self.run_frontier_mode()
            return

        if self.state == 'GO_TO_STATIONARY':
            # Once the stationary tin is in the release window for N cycles, dispense.
            if self.update_release_counter(stationary_tag_id):
                if self.nav_busy:
                    self.cancel_current_goal()
                self.trigger_dispense()
                self.state = 'EXPLORE'
                return

            # If stationary tag disappears for too long, go back to exploration.
            if self.get_fresh_detection(stationary_tag_id) is None and not self.nav_busy:
                self.get_logger().info('Stationary tag lost. Returning to explore.')
                self.state = 'EXPLORE'
                self.release_frame_counter = 0
                return

            return

        if self.state == 'GO_TO_MIDPOINT':
            # Wait for nav result callback to move us to WAIT_FOR_MOVING_TIN.
            # If midpoint tag is lost and nav is idle, fall back to exploration.
            if self.get_fresh_detection(midpoint_tag_id) is None and not self.nav_busy:
                self.get_logger().info('Midpoint tag lost. Returning to explore.')
                self.state = 'EXPLORE'
                self.release_frame_counter = 0
            return

        if self.state == 'WAIT_FOR_MOVING_TIN':
            if self.update_release_counter(moving_tag_id):
                self.trigger_dispense()
                self.state = 'EXPLORE'
                return

            # If midpoint and moving tin are both gone, resume exploring.
            if (
                self.get_fresh_detection(midpoint_tag_id) is None and
                self.get_fresh_detection(moving_tag_id) is None
            ):
                self.get_logger().info('Moving-tin tracking lost. Returning to explore.')
                self.state = 'EXPLORE'
                self.release_frame_counter = 0
            return


def main(args=None):
    rclpy.init(args=args)
    node = Coordinator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()