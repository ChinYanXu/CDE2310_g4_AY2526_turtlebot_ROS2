#main code for AY2526 CDE2310 attempt

#!/usr/bin/env python3

# adapted from https://github.com/SeanReg/nav2_wavefront_frontier_exploration

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# If your AprilTag package uses a different message, change this import and callback only.
# from apriltag_msgs.msg import AprilTagDetectionArray

from cde2310_g4_AY2526.frontier_detection import (
    OccupancyGrid2d,
    detect_frontiers,
    choose_frontier,
)

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer_main')

        self.declare_parameter('map_topic', '/map')
        # self.declare_parameter('tag_topic', '/tag_detections')
        # self.declare_parameter('target_tag_id', -1)   # -1 = any detected tag interrupts navigation
        self.declare_parameter('planner_period_sec', 1.0)
        self.declare_parameter('min_frontier_size', 5)
        self.declare_parameter('frontier_strategy', 'nearest')

        map_topic = self.get_parameter('map_topic').value
        # tag_topic = self.get_parameter('tag_topic').value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.map_callback,
            10
        )

        # self.tag_sub = self.create_subscription(
        #     AprilTagDetectionArray,
        #     tag_topic,
        #     self.tag_callback,
        #     10
        # )

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(
            float(self.get_parameter('planner_period_sec').value),
            self.control_loop
        )

        self.map_msg = None
        self.goal_handle = None
        self.nav_busy = False
        # self.tag_detected = False
        self.exploration_done = False

        self.get_logger().info('Frontier explorer main node started.')

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg

    # def tag_callback(self, msg: AprilTagDetectionArray):
    #     target_tag_id = int(self.get_parameter('target_tag_id').value)

    #     valid = False
    #     for det in msg.detections:
    #         if len(det.id) == 0:
    #             continue
    #         if target_tag_id == -1 or int(det.id[0]) == target_tag_id:
    #             valid = True
    #             break

    #     if valid:
    #         if not self.tag_detected:
    #             self.get_logger().info('Valid AprilTag detected. Interrupting frontier navigation.')
    #         self.tag_detected = True
    #         self.cancel_current_goal()

    def get_robot_pose_in_map(self):
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

    def control_loop(self):
        if self.exploration_done:
            return

        # # If AprilTag has been detected, stop exploration.
        # if self.tag_detected:
        #     return

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

        if not frontiers:
            self.get_logger().info('No more frontiers found. Exploration complete.')
            self.exploration_done = True
            return

        strategy = self.get_parameter('frontier_strategy').value
        chosen = choose_frontier(frontiers, robot_pose_stamped.pose, strategy=strategy)

        if chosen is None:
            self.get_logger().warn('Frontier detection returned no valid goal.')
            return

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = chosen.x
        goal.pose.position.y = chosen.y
        goal.pose.position.z = 0.0

        # Use current robot orientation to avoid unnecessary spin at frontier goal.
        goal.pose.orientation = robot_pose_stamped.pose.orientation

        self.get_logger().info(
            f'Sending frontier goal x={goal.pose.position.x:.2f}, y={goal.pose.position.y:.2f}, size={chosen.size}'
        )
        self.send_nav_goal(goal)

    def send_nav_goal(self, goal_pose: PoseStamped):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('NavigateToPose action server not available yet.')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.nav_busy = True
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('Frontier goal rejected.')
            self.nav_busy = False
            return

        self.goal_handle = goal_handle
        self.get_logger().info('Frontier goal accepted.')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        self.nav_busy = False

        result = future.result()
        if result is None:
            self.get_logger().warn('No result received from NavigateToPose.')
            return

        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Frontier goal reached.')
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Frontier goal canceled.')
        else:
            self.get_logger().warn(f'Frontier goal ended with status {status}.')

    def cancel_current_goal(self):
        if self.goal_handle is None:
            return
        if not self.nav_busy:
            return

        self.get_logger().info('Canceling current frontier goal...')
        cancel_future = self.goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        self.get_logger().info('Cancel request sent to NavigateToPose.')

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()