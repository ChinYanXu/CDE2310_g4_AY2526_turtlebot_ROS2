#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from opennav_docking_msgs.action import DockRobot, UndockRobot
from action_msgs.msg import GoalStatus
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

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
        self.last_sent_goal: Optional[PoseStamped] = None
        # frontier detection params
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('planner_period_sec', 0.1)
        self.declare_parameter('min_frontier_size', 3)
        self.declare_parameter('frontier_strategy', 'largest')
        self.declare_parameter('fallback_min_obstacle_clearance_cells', 2)
        self.declare_parameter('fallback_revisit_radius_m', 0.8)
        self.declare_parameter('max_recent_fallbacks', 15)
        self.debug_broadcaster = TransformBroadcaster(self)
        
        # aruco tag ids
        self.declare_parameter('stationary_tag_id', 1)
        self.declare_parameter('midpoint_tag_id', 2)
        self.declare_parameter('moving_tag_id', 3) # handled by payload node

        self.declare_parameter('detection_timeout_sec', 2)

        # dispenser trigger
        self.declare_parameter('dispense_topic', '/station_cmd')

        map_topic = self.get_parameter('map_topic').value
        self.dispense_topic = self.get_parameter('dispense_topic').value

        self.pending_dock_goal: Optional[str] = None
        self.last_known_dock_pose: Optional[PoseStamped] = None

        # ===== Subscriptions =====

        # FIX 1: map uses TRANSIENT_LOCAL so a late subscriber still gets the last message
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, map_qos
        )

        self.aruco_left_sub = self.create_subscription(
            ArucoMarkers, '/cam_left/aruco_markers', self.aruco_left_cb, 10
        )

        # ===== Publishers / actions =====
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.dispense_pub = self.create_publisher(String, self.dispense_topic, 10)

        self.detected_dock_pose_pub = self.create_publisher(
            PoseStamped, '/detected_dock_pose', 10
        )

        self.dock_client = ActionClient(self, DockRobot, '/dock_robot')
        self.undock_client = ActionClient(self, UndockRobot, '/undock_robot')        

        # ===== TF =====
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ===== Timer =====
        self.timer = self.create_timer(
            float(self.get_parameter('planner_period_sec').value),
            self.control_loop
        )

        self.dock_pose_timer = self.create_timer(0.1, self.dock_pose_loop)
        self.active_dock_tag_id: Optional[int] = None

        # ===== State =====
        self.map_msg: Optional[OccupancyGrid] = None
        self.goal_handle = None
        self.nav_busy = False
        self.last_goal_type: Optional[str] = None
        self.recent_fallback_points = []
        self.state_entry_time = 0.0
        self.mission_timeout = 90.0
        self.state = 'EXPLORE'
        # EXPLORE, GO_TO_STATIONARY, GO_TO_MIDPOINT, WAIT_A_COMPLETE, WAIT_B_COMPLETE

        self.latest_detections: Dict[int, DetectionRecord] = {}
        self.latest_tag_poses: Dict[int, PoseStamped] = {}

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
        #self.get_logger().info('Map received.')  # log once so you know QoS is working

    def aruco_left_cb(self, msg: ArucoMarkers):
        self.handle_aruco_msg(msg, 'cam_left')

    def handle_aruco_msg(self, msg: ArucoMarkers, source: str):
        now_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            marker_id = int(marker_id)
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = pose

            new_record = DetectionRecord(
                frame_id=msg.header.frame_id,
                stamp_sec=now_sec,
                marker_id=marker_id,
                pose_stamped=ps,
                source=source,
            )

            old = self.latest_detections.get(marker_id)
            if old is None:
                self.latest_detections[marker_id] = new_record
                self.update_tag_pose_in_map(marker_id, new_record)
                continue

            if new_record.stamp_sec > old.stamp_sec:
                self.latest_detections[marker_id] = new_record
                self.update_tag_pose_in_map(marker_id, new_record)

    # ------------------------------------------------------------------
    # Time / TF helpers
    # ------------------------------------------------------------------

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def get_robot_pose_in_map(self) -> Optional[PoseStamped]:
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=0.3)
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
        now_sec = self.get_clock().now().nanoseconds / 1e9
        age = now_sec - rec.stamp_sec
        timeout_sec = float(self.get_parameter('detection_timeout_sec').value)
        if age > timeout_sec:
            return None
        # self.get_logger().info(f'[FRESH] now={now_sec:.3f} rec={rec.stamp_sec:.3f} age={age:.3f}') 
        return rec

    def get_tag_pose_in_map(self, marker_id: int) -> Optional[PoseStamped]:
        rec = self.get_fresh_detection(marker_id)
        if rec is None:
            return None

        pose_in_map = self.latest_tag_poses.get(marker_id)
        if pose_in_map is None:
            self.get_logger().warn(f'No cached map pose for tag {marker_id}.')
            return None

        return pose_in_map

    def update_tag_pose_in_map(self, marker_id: int, rec: DetectionRecord):
        target_frame = 'odom'
        detection_frame = rec.pose_stamped.header.frame_id
        if not self.tf_buffer.can_transform(target_frame, detection_frame, rclpy.time.Time()):
            self.get_logger().warn(
                f'{detection_frame} not yet connected to {target_frame}, skipping.'
            )
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                detection_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )

            raw = rec.pose_stamped.pose.position
            # Remap optical frame -> ROS camera_link convention before applying TF.
            remapped_pose = PoseStamped()
            remapped_pose.header = rec.pose_stamped.header
            remapped_pose.pose = Pose()
            remapped_pose.pose.orientation = rec.pose_stamped.pose.orientation
            remapped_pose.pose.position.x = raw.z
            remapped_pose.pose.position.y = -raw.x
            remapped_pose.pose.position.z = -raw.y

            transformed_pose = do_transform_pose(remapped_pose.pose, tf)

            # Rotate optical Z (0,0,1) by tag quaternion to get normal in optical frame
            q = rec.pose_stamped.pose.orientation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            nx = 2.0 * (qx*qz + qw*qy)
            ny = 2.0 * (qy*qz - qw*qx)
            nz = 1.0 - 2.0 * (qx*qx + qy*qy)

            # Remap optical -> ROS (same as position remap)
            nx_ros, ny_ros, nz_ros = nz, -nx, -ny

            # Rotate by TF quaternion to get normal in odom frame
            tf_q = tf.transform.rotation
            tx, ty, tz, tw = tf_q.x, tf_q.y, tf_q.z, tf_q.w
            cx = ty*nz_ros - tz*ny_ros
            cy = tz*nx_ros - tx*nz_ros
            cz = tx*ny_ros - ty*nx_ros
            rx = nx_ros + 2.0*tw*cx + 2.0*(ty*cz - tz*cy)
            ry = ny_ros + 2.0*tw*cy + 2.0*(tz*cx - tx*cz)
            yaw = math.atan2(ry, rx)

            result = PoseStamped()

            result.header.frame_id = target_frame
            result.header.stamp = self.get_clock().now().to_msg()
            result.pose.position = transformed_pose.position
            result.pose.orientation.x = 0.0
            result.pose.orientation.y = 0.0
            result.pose.orientation.z = math.sin(yaw / 2.0)
            result.pose.orientation.w = math.cos(yaw / 2.0)
            self.latest_tag_poses[marker_id] = result

            debug_tf = TransformStamped()
            debug_tf.header.stamp = result.header.stamp
            debug_tf.header.frame_id = target_frame
            debug_tf.child_frame_id = f'debug_tag_{marker_id}'
            debug_tf.transform.translation.x = result.pose.position.x
            debug_tf.transform.translation.y = result.pose.position.y
            debug_tf.transform.translation.z = result.pose.position.z
            debug_tf.transform.rotation = result.pose.orientation
            self.debug_broadcaster.sendTransform(debug_tf)
        except Exception as ex:
            self.get_logger().warn(
                f'Could not transform tag {marker_id} from {detection_frame} '
                f'to {target_frame}: {ex}'
            )

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

    # ------------------------------------------------------------------
    # Nav2 / docking helpers
    # ------------------------------------------------------------------

    def publish_dock_pose(self, marker_id: int):
        pose = self.get_tag_pose_in_map(marker_id)

        if pose is not None:
            # Fresh detection — update latch and reset lost timer
            self.last_known_dock_pose = pose

        if self.state in ('GO_TO_MIDPOINT', 'GO_TO_STATIONARY'):
            if self.last_known_dock_pose is None:
                return
            out = PoseStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = self.last_known_dock_pose.header.frame_id
            out.pose = self.last_known_dock_pose.pose
            self.detected_dock_pose_pub.publish(out)

    def send_dock_goal(self, dock_id: str):
        if not self.dock_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('DockRobot action server not available.')
            return False

        goal = DockRobot.Goal()
        goal.dock_id = dock_id          # matches the ID in your docks YAML
        goal.max_staging_time = 10.0
        goal.navigate_to_staging_pose = False  # let Nav2 drive to vicinity first

        self.nav_busy = True
        future = self.dock_client.send_goal_async(goal)
        future.add_done_callback(self.dock_response_callback)
        return True

    def send_undock_goal(self):
        if not self.undock_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('UndockRobot action server not available.')
            return False

        goal = UndockRobot.Goal()
        goal.dock_type = 'aruco_dock'
        # goal.max_undocking_time = 15.0

        self.nav_busy = True
        future = self.undock_client.send_goal_async(goal)
        future.add_done_callback(self.undock_response_callback)
        return True

    def dock_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Dock goal rejected.')
            self.nav_busy = False
            return
        goal_handle.get_result_async().add_done_callback(self.dock_result_callback)

    def undock_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Undock goal rejected.')
            self.nav_busy = False
            return
        goal_handle.get_result_async().add_done_callback(self.undock_result_callback)

    def dock_result_callback(self, future):
        self.nav_busy = False
        self.active_dock_tag_id = None
        self.last_known_dock_pose = None

        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Docking succeeded.')
            if self.last_goal_type == 'stationary':
                self.trigger_dispense("A")
                self.state = 'WAIT_A_COMPLETE'
            elif self.last_goal_type == 'midpoint':
                self.trigger_dispense("B")
                self.state = 'WAIT_B_COMPLETE'
        else:
            self.get_logger().warn(f'Docking failed with status {status}.')
            self.state = 'EXPLORE'

    def undock_result_callback(self, future):
        self.nav_busy = False
        self.active_dock_tag_id = None

        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Undocking succeeded.')
        else:
            self.get_logger().warn(f'Undocking failed with status {status}.')

        self.state = 'EXPLORE'

    def dock_pose_loop(self):
        if self.active_dock_tag_id is None:
            return
        self.publish_dock_pose(self.active_dock_tag_id)

    def send_nav_goal(self, goal_pose: PoseStamped):
        self.last_sent_goal = goal_pose
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
            self.get_logger().warn(
                f'Goal rejected. '
                f'frame={self.last_sent_goal.header.frame_id} '
                f'x={self.last_sent_goal.pose.position.x:.3f} '
                f'y={self.last_sent_goal.pose.position.y:.3f}'
            )
            self.nav_busy = False
            self.goal_handle = None
            return

        self.goal_handle = goal_handle
        self.get_logger().info('Goal accepted.')
        goal_handle.get_result_async().add_done_callback(self.goal_result_callback)

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

        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Goal canceled.')
        else:
            self.get_logger().warn(f'Goal ended with status {status}. Continuing exploration.')
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
        self.get_logger().info('Cancel request completed.')

        self.nav_busy = False

        if self.pending_dock_goal is not None:
            dock_id = self.pending_dock_goal
            self.pending_dock_goal = None

            self.get_logger().info(f'Sending dock goal after cancel: {dock_id}')

            if self.send_dock_goal(dock_id):
                if dock_id == 'stationary_dock':
                    self.state = 'GO_TO_STATIONARY'
                    self.state_entry_time = self.now_sec()
                elif dock_id == 'midpoint_dock':
                    self.state = 'GO_TO_MIDPOINT'
                    self.state_entry_time = self.now_sec()
            else:
                self.nav_busy = False

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
        msg = String()
        if mode == "A":
            msg.data = "START_A"
        elif mode == "B":
            msg.data = "START_B"
        else:
            self.get_logger().error(f"Invalid mode: {mode}")
            return

        self.dispense_pub.publish(msg)
        self.state_entry_time = self.now_sec()
        self.get_logger().info(f'DISPENSE TRIGGERED: {msg.data}')

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
        chosen_frontier = choose_frontier(frontiers, robot_pose_stamped.pose, strategy=strategy)

        if chosen_frontier is not None:
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = chosen_frontier.x
            goal.pose.position.y = chosen_frontier.y
            goal.pose.position.z = 0.0
            dx = chosen_frontier.x - robot_pose_stamped.pose.position.x
            dy = chosen_frontier.y - robot_pose_stamped.pose.position.y
            goal.pose.orientation = self.yaw_to_quaternion(math.atan2(dy, dx))
            self.get_logger().info(
                f'[Frontier] Sending goal x={goal.pose.position.x:.2f} '
                f'y={goal.pose.position.y:.2f} size={chosen_frontier.size}'
            )
            self.last_goal_type = 'frontier'
            self.send_nav_goal(goal)
            return

        fallback = choose_fallback_viewpoint(
            costmap=costmap,
            robot_pose=robot_pose_stamped.pose,
            recent_points=self.recent_fallback_points,
            min_clearance_cells=int(self.get_parameter('fallback_min_obstacle_clearance_cells').value),
            revisit_radius=float(self.get_parameter('fallback_revisit_radius_m').value),
        )

        if fallback is None:
            self.get_logger().warn('No fallback viewpoint found.')
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
        goal.pose.orientation = self.yaw_to_quaternion(math.atan2(dy, dx))
        self.get_logger().info(f'[Fallback] Sending goal x={fx:.2f} y={fy:.2f}')
        self.last_goal_type = 'fallback'
        self.send_nav_goal(goal)

    # ------------------------------------------------------------------
    # Main coordinator state machine
    # ------------------------------------------------------------------

    def control_loop(self):
        stationary_tag_id = int(self.get_parameter('stationary_tag_id').value)
        midpoint_tag_id   = int(self.get_parameter('midpoint_tag_id').value)

        if self.map_msg is None:
            self.get_logger().info('Waiting for map...', throttle_duration_sec=5.0)
            return

        if self.state in ['WAIT_A_COMPLETE', 'WAIT_B_COMPLETE', 'GO_TO_STATIONARY', 'GO_TO_MIDPOINT']:
            if (self.now_sec() - self.state_entry_time) > self.mission_timeout:
                self.get_logger().error(f'TIMEOUT waiting for {self.state}. Resuming exploration.')
                if self.state == 'WAIT_A_COMPLETE': self.a_complete = True
                if self.state == 'WAIT_B_COMPLETE': self.b_complete = True
                self.state = 'EXPLORE'

        elif self.state == 'EXPLORE':
            stationary_pose_map = self.get_tag_pose_in_map(stationary_tag_id)
            if stationary_pose_map is not None and not self.a_complete:
                self.last_goal_type = 'stationary'
                self.active_dock_tag_id = stationary_tag_id
                self.get_logger().info('Stationary tag detected. Sending dock goal...')
                if self.nav_busy:
                    self.pending_dock_goal = 'stationary_dock'
                    self.cancel_current_goal()
                    return
                if self.send_dock_goal('stationary_dock'):
                    self.state = 'GO_TO_STATIONARY'
                    self.state_entry_time = self.now_sec()
                    return

            midpoint_pose_map = self.get_tag_pose_in_map(midpoint_tag_id)
            if midpoint_pose_map is not None and not self.b_complete:
                self.last_goal_type = 'midpoint'
                self.active_dock_tag_id = midpoint_tag_id
                self.get_logger().info('Midpoint tag detected. Sending dock goal...')
                if self.nav_busy:
                    self.pending_dock_goal = 'midpoint_dock'
                    self.cancel_current_goal()
                    return
                if self.send_dock_goal('midpoint_dock'):
                    self.state = 'GO_TO_MIDPOINT'
                    self.state_entry_time = self.now_sec()
                    return
                
            self.run_frontier_mode()
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