#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from geometry_msgs.msg import TransformStamped    # [J] tf broadcasting for debug
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap, ClearCostmapAroundRobot
from opennav_docking_msgs.action import DockRobot, UndockRobot
from action_msgs.msg import GoalStatus
from tf2_ros import TransformBroadcaster  # [J] tf broadcasting for debug
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
        self.last_sent_goal: Optional[PoseStamped] = None    # [J] last_sent_goal not defined, define it first
        # frontier detection params
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('planner_period_sec', 0.1)
        self.declare_parameter('min_frontier_size', 3)
        self.declare_parameter('frontier_strategy', 'largest')
        self.declare_parameter('fallback_min_obstacle_clearance_cells', 2)
        self.declare_parameter('fallback_revisit_radius_m', 0.8)
        self.declare_parameter('max_recent_fallbacks', 15)
        self.debug_static_broadcaster = TransformBroadcaster(self)   # [J] static broadcaster
        # aruco tag and offset params
        self.declare_parameter('stationary_tag_id', 1)
        self.declare_parameter('moving_tag_id', 2)
        self.declare_parameter('midpoint_tag_id', 3)

        self.declare_parameter('goal_standoff_m', 0.15)       # [J] modified goal standoff
        self.declare_parameter('midpoint_standoff_m', 0.15)   # [J] might change to not just a standoff

        self.declare_parameter('detection_timeout_sec', 2)    # [J] this is for high latency

        # Camera-frame release gate.
        # Adjust axes/thresholds after testing.
        self.declare_parameter('release_lateral_threshold_m', 0.05)
        self.declare_parameter('release_forward_min_m', 0.15)
        self.declare_parameter('release_forward_max_m', 0.32)
        self.declare_parameter('release_required_consecutive_frames', 3)

        # Which camera topic should be trusted for release checks.
        self.declare_parameter('release_camera_source', 'rpi')

        # Optional topic to trigger dispenser.
        self.declare_parameter('dispense_topic', '/station_cmd')

        map_topic = self.get_parameter('map_topic').value
        self.dispense_topic = self.get_parameter('dispense_topic').value

        self.last_known_dock_pose: Optional[PoseStamped] = None
        self.dock_pose_lost_time: Optional[float] = None
        self.declare_parameter('dock_pose_latch_timeout_sec', 5.0)  # how long to hold last known pose

        # ===== Subscriptions =====
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, 10
        )

        self.aruco_left_sub = self.create_subscription(
            ArucoMarkers, '/cam_left/aruco_markers', self.aruco_left_cb, 10
        )
        # self.aruco_right_sub = self.create_subscription(
        #     ArucoMarkers, '/cam_right/aruco_markers', self.aruco_right_cb, 10
        # )
        # self.aruco_rpi_sub = self.create_subscription(
        #     ArucoMarkers, '/rpi/aruco_markers', self.aruco_rpi_cb, 10
        # )

        # ===== Publishers / actions =====
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.dispense_pub = self.create_publisher(String, self.dispense_topic, 10)
        # self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.detected_dock_pose_pub = self.create_publisher(
            PoseStamped, '/detected_dock_pose', 10
        )

        self.dock_client = ActionClient(self, DockRobot, '/dock_robot')

        # Costmap clearing service for docking
        self.clear_local_costmap_client = self.create_client(
            ClearCostmapAroundRobot, '/local_costmap/clear_entirely_local_costmap'
        )
        self.clear_global_costmap_client = self.create_client(
            ClearCostmapAroundRobot, '/global_costmap/clear_entirely_global_costmap'
        )

        # ===== TF =====
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ===== Timer =====
        self.timer = self.create_timer(
            float(self.get_parameter('planner_period_sec').value),
            self.control_loop
        )

        self.dock_pose_timer = self.create_timer(0.1, self.dock_pose_loop)  # 10 Hz
        self.active_dock_tag_id: Optional[int] = None

        # ===== State =====
        self.map_msg: Optional[OccupancyGrid] = None
        self.goal_handle = None
        self.nav_busy = False
        self.last_goal_type: Optional[str] = None
        self.recent_fallback_points = []
        self.state_entry_time = 0.0      # <--- ADD THIS
        self.mission_timeout = 50.0       # <--- ADD THIS
        self.state = 'EXPLORE'
        # EXPLORE, GO_TO_STATIONARY, GO_TO_MIDPOINT, WAIT_FOR_MOVING_TIN

        self.latest_detections: Dict[int, DetectionRecord] = {}
        self.release_frame_counter = 0
        self.last_dispense_time_sec = -999.0

        self.a_complete = False
        self.b_complete = False
        self.mission_complete_sub = self.create_subscription(
            String, '/mission_complete', self.mission_complete_cb, 10
        )
        self.get_logger().info('Coordinator node started.')

    # ------------------------------------------------------------------
    # Basic callbacks
    # ------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg

    def aruco_left_cb(self, msg: ArucoMarkers):
        self.handle_aruco_msg(msg, 'cam_left')

    # def aruco_right_cb(self, msg: ArucoMarkers):
    #     self.handle_aruco_msg(msg, 'cam_right')

    # def aruco_rpi_cb(self, msg: ArucoMarkers):
    #     self.handle_aruco_msg(msg, 'rpi')

    def handle_aruco_msg(self, msg: ArucoMarkers, source: str):
        now_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9       # [J] new now_Sec for more consistent timing
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
        now_sec = self.get_clock().now().nanoseconds / 1e9   # [J] for more accurate now sec
        age = now_sec - rec.stamp_sec  # [J] tag age
        timeout_sec = float(self.get_parameter('detection_timeout_sec').value)
        if age > timeout_sec:#only fresh tags
            return None
        self.get_logger().info(f'[FRESH] now={now_sec:.3f} rec={rec.stamp_sec:.3f} age={age:.3f}') #debug
        return rec

    def get_tag_pose_in_map(self, marker_id: int) -> Optional[PoseStamped]:
        rec = self.get_fresh_detection(marker_id)
        if rec is None:
            # self.get_logger().info("no tags detected")
            return None
        # else:
            # pose = rec.pose_stamped
            # return pose

        if not self.tf_buffer.can_transform('map', 'cam_left_camera_link', rclpy.time.Time()):
            self.get_logger().warn('cam_left_camera_link not yet connected to map, skipping.')   #during startup of coordinator the tf trees arent connected yet
            return None
        self.get_logger().info(f'[TF] Attempting transform for tag {marker_id} from {rec.pose_stamped.header.frame_id}') #debug
        try:
            tf = self.tf_buffer.lookup_transform(
                'map',
                rec.pose_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)  #might have to change this
            )

            #from here onwards is for debugging by showing the tf on rviz
            transformed_pose = do_transform_pose(rec.pose_stamped.pose, tf)  
            self.get_logger().info(f'[TF] Tag {marker_id} transformed successfully')
            result = PoseStamped()
            result.header.frame_id = 'map'
            result.header.stamp = self.get_clock().now().to_msg()
            result.pose = transformed_pose
            # Override orientation to point from tag toward robot
            # robot_pose = self.get_robot_pose_in_map()
            # if robot_pose is not None:
            #     dx = robot_pose.pose.position.x - result.pose.position.x
            #     dy = robot_pose.pose.position.y - result.pose.position.y
            #     yaw = math.atan2(dy, dx)
            #     result.pose.orientation = self.yaw_to_quaternion(yaw)
            # Debug TF
            from geometry_msgs.msg import TransformStamped
            debug_tf = TransformStamped()
            debug_tf.header.stamp = self.get_clock().now().to_msg()
            debug_tf.header.frame_id = 'map'
            debug_tf.child_frame_id = f'debug_tag_{marker_id}'
            debug_tf.transform.translation.x = result.pose.position.x
            debug_tf.transform.translation.y = result.pose.position.y
            debug_tf.transform.translation.z = result.pose.position.z
            debug_tf.transform.rotation = result.pose.orientation
            self.debug_static_broadcaster.sendTransform(debug_tf)
            return result 
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

    def publish_dock_pose(self, marker_id: int):
        pose = self.get_tag_pose_in_map(marker_id)

        if pose is not None:
            # Fresh detection — update latch and reset lost timer
            self.last_known_dock_pose = pose
            self.dock_pose_lost_time = None

        # Publish either fresh or latched pose
        if self.state == 'GO_TO_MIDPOINT' or self.state == 'GO_TO_STATIONARY':
            out = PoseStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = 'map'
            out.pose = self.last_known_dock_pose.pose
            self.detected_dock_pose_pub.publish(out)

    def send_dock_goal(self, dock_id: str):
        if not self.dock_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('DockRobot action server not available.')
            return False

        goal = DockRobot.Goal()
        goal.dock_id = dock_id          # matches the ID in your docks YAML
        goal.max_staging_time = 10.0
        goal.navigate_to_staging_pose = True  # let Nav2 drive to vicinity first

        self.nav_busy = True
        future = self.dock_client.send_goal_async(goal)
        future.add_done_callback(self.dock_response_callback)
        return True

    def send_undock_goal(self):
        goal = UndockRobot.Goal()
        goal.dock_type = 'aruco_dock'
        goal.max_undocking_time = 2.0

        self.nav_busy = True
        future = self.dock_client.send_goal_async(goal)
        future.add_done_callback(self.dock_response_callback)
        return True

    def dock_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Dock goal rejected.')
            self.nav_busy = False
            return
        goal_handle.get_result_async().add_done_callback(self.dock_result_callback)

    def dock_result_callback(self, future):
        self.nav_busy = False
        self.active_dock_tag_id = None
        self.last_known_dock_pose = None   # clear latch
        self.dock_pose_lost_time = None
        
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Docking succeeded.')
            if self.last_goal_type == 'stationary':
                self.trigger_dispense("A")
                self.state = 'WAIT_A_COMPLETE'
            elif self.last_goal_type == 'midpoint':
                self.state = 'WAIT_FOR_MOVING_TIN'
        else:
            self.get_logger().warn(f'Docking failed with status {status}.')
            self.state = 'EXPLORE'

    def dock_pose_loop(self):
        if self.active_dock_tag_id is None:
            return
        self.publish_dock_pose(self.active_dock_tag_id)

    def send_nav_goal(self, goal_pose: PoseStamped):
        self.last_sent_goal = goal_pose   #fixed last_sent goal not defined
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
            self.get_logger().warn(f'Goal rejected. Handle={goal_handle}') #more verbose
            self.get_logger().warn(
                f'Rejected goal was: frame={self.last_sent_goal.header.frame_id} '
                f'x={self.last_sent_goal.pose.position.x:.3f} '
                f'y={self.last_sent_goal.pose.position.y:.3f}'
            )
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

    def mission_complete_cb(self, msg):
        if msg.data == 'FINISH_A':
            self.get_logger().info('Station A complete. Undocking and resuming explore.')
            self.a_complete = True
            self.send_undock_goal()
            self.state = 'EXPLORE'
        elif msg.data == 'FINISH_B':
            self.get_logger().info('Station B complete. Undocking and resuming explore.')
            self.b_complete = True
            self.send_undock_goal()
            self.state = 'EXPLORE'
    # ------------------------------------------------------------------
    # Dispense helpers
    # ------------------------------------------------------------------

    def trigger_dispense(self, mode): 
        # mode should be "A" or "B"
        from std_msgs.msg import String

        msg = String()

        if mode == "A":
            msg.data = "START_A"
        elif mode == "B":
            msg.data = "START_B"
        else:
            self.get_logger().error(f"Invalid mode: {mode}")
            return

        self.dispense_pub.publish(msg)
        self.last_dispense_time_sec = self.now_sec()
        self.state_entry_time = self.now_sec()  # <--- ADD THIS
        self.release_frame_counter = 0
        self.get_logger().info(f'DISPENSE TRIGGERED: {msg.data}')

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
            self.get_logger().info('Sending frontier goal')
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
        self.get_logger().info('Sending fallback goal')
        self.send_nav_goal(goal)

    # ------------------------------------------------------------------
    # Main coordinator state machine
    # ------------------------------------------------------------------

    def control_loop(self):
        if self.map_msg is None:
            return
        if self.state in ['WAIT_A_COMPLETE', 'WAIT_B_COMPLETE']:
            if (self.now_sec() - self.state_entry_time) > self.mission_timeout:
                self.get_logger().error(f'TIMEOUT waiting for {self.state}. Resuming exploration.')
                if self.state == 'WAIT_A_COMPLETE': self.a_complete = True
                if self.state == 'WAIT_B_COMPLETE': self.b_complete = True
                self.state = 'EXPLORE'
            return 
        
        stationary_tag_id = int(self.get_parameter('stationary_tag_id').value)
        moving_tag_id = int(self.get_parameter('moving_tag_id').value)
        midpoint_tag_id = int(self.get_parameter('midpoint_tag_id').value)

        # Priority:
        # 1) stationary tin if visible
        # 2) midpoint for moving tin
        # 3) exploration

        if self.state == 'EXPLORE':
            stationary_pose_map = self.get_tag_pose_in_map(stationary_tag_id)
            if stationary_pose_map is not None and not self.a_complete:
                if self.nav_busy:
                    self.cancel_current_goal()
                self.last_goal_type = 'stationary'
                self.active_dock_tag_id = stationary_tag_id
                if self.send_dock_goal('stationary_dock'):
                    self.state = 'GO_TO_STATIONARY'
                    return

            midpoint_pose_map = self.get_tag_pose_in_map(midpoint_tag_id)
            if midpoint_pose_map is not None and not self.b_complete:
                if self.nav_busy:
                    self.cancel_current_goal()
                self.last_goal_type = 'midpoint'
                self.active_dock_tag_id = midpoint_tag_id
                if self.send_dock_goal('midpoint_dock'):
                    self.state = 'GO_TO_MIDPOINT'
                    return
            self.run_frontier_mode()
            return

        if self.state == 'GO_TO_STATIONARY':
            # Once the stationary tin is in the release window for N cycles, dispense.
            if self.update_release_counter(stationary_tag_id):
                if self.nav_busy:
                    self.cancel_current_goal()
                self.trigger_dispense("A")
                self.state = 'WAIT_A_COMPLETE'
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
                self.trigger_dispense("B")
                self.state = 'WAIT_B_COMPLETE'
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
        
        if self.state == 'WAIT_B_COMPLETE':
            if (
                 self.get_fresh_detection(midpoint_tag_id) is None and
                 self.get_fresh_detection(moving_tag_id) is None
            ):
                 self.get_logger().warn('Tags lost while waiting for FINISH_B. Resuming explore.')
                 self.state = 'EXPLORE'
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