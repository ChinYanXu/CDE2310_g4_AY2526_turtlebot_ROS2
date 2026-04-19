#!/usr/bin/env python3
# adapted from wavefront frontier exploration style,
# with fallback viewpoint selection when no reachable frontier is found

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

from cde2310_g4_ay2526.frontier_detection import (
    OccupancyGrid2d,
    detect_frontiers,
    choose_frontier,
    choose_fallback_viewpoint,
)


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer_main')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('planner_period_sec', 1.0)
        self.declare_parameter('min_frontier_size', 3)
        self.declare_parameter('frontier_strategy', 'largest')
        self.declare_parameter('fallback_min_obstacle_clearance_cells', 2)
        self.declare_parameter('fallback_revisit_radius_m', 0.8)
        self.declare_parameter('max_recent_fallbacks', 15)

        map_topic = self.get_parameter('map_topic').value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.map_callback,
            10
        )

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
        self.exploration_done = False

        self.no_frontier_count = 0
        self.failed_goal_count = 0

        self.last_goal_type = None
        self.recent_fallback_points = []

        self.get_logger().info('Frontier explorer main node started.')

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg

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

        if self.nav_busy:
            return

        if self.map_msg is None:
            self.get_logger().info('Waiting for map...')
            return

        robot_pose_stamped = self.get_robot_pose_in_map()
        if robot_pose_stamped is None:
            return

        costmap = OccupancyGrid2d(self.map_msg)

        data = list(self.map_msg.data)
        free_count = sum(1 for c in data if c == 0)
        unknown_count = sum(1 for c in data if c == -1)
        occupied_count = sum(1 for c in data if c > 0)

        self.get_logger().info(
            f'Map stats: free={free_count}, unknown={unknown_count}, occupied={occupied_count}'
        )

        try:
            mx, my = costmap.world_to_map(
                robot_pose_stamped.pose.position.x,
                robot_pose_stamped.pose.position.y
            )
            self.get_logger().info(
                f'Robot map cell=({mx}, {my}), cost={costmap.get_cost(mx, my)}'
            )
        except ValueError:
            self.get_logger().warn('Robot pose is outside map bounds.')
            return

        frontiers = detect_frontiers(
            costmap,
            robot_pose_stamped.pose,
            min_frontier_size=int(self.get_parameter('min_frontier_size').value)
        )

        self.get_logger().info(f'Detected {len(frontiers)} frontiers.')

        strategy = self.get_parameter('frontier_strategy').value
        chosen_frontier = choose_frontier(
            frontiers,
            robot_pose_stamped.pose,
            strategy=strategy
        )

        if chosen_frontier is not None:
            self.no_frontier_count = 0

            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = chosen_frontier.x
            goal.pose.position.y = chosen_frontier.y
            goal.pose.position.z = 0.0
            goal.pose.orientation = robot_pose_stamped.pose.orientation

            self.get_logger().info(
                f'Sending frontier goal '
                f'x={goal.pose.position.x:.2f}, y={goal.pose.position.y:.2f}, '
                f'size={chosen_frontier.size}'
            )

            self.last_goal_type = 'frontier'
            self.send_nav_goal(goal)
            return

        self.no_frontier_count += 1
        self.get_logger().info(
            f'No usable frontier this cycle. '
            f'Fallback attempt {self.no_frontier_count}.'
        )

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
        goal.pose.orientation = robot_pose_stamped.pose.orientation

        self.get_logger().info(
            f'[Fallback viewpoint] Sending goal x={fx:.2f}, y={fy:.2f}'
        )

        self.last_goal_type = 'fallback'
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
            self.get_logger().warn('Goal rejected.')
            self.nav_busy = False
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
            self.get_logger().warn(f'Goal ended with status {status}.')

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