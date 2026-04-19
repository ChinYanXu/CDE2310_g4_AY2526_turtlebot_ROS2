from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # Aruco Node - Cam Left
        Node(
            package='ros2_aruco',
            executable='aruco_node',
            name='aruco_node',
            namespace='cam_left',
            parameters=[{
                'marker_size': 0.092,
                'aruco_dictionary_id': 'DICT_4X4_50',
                'image_topic': '/cam_left/image_raw',
                'camera_info_topic': '/cam_left/camera_info',
            }]
        ),

        # Aruco Node - Cam Right
        Node(
            package='ros2_aruco',
            executable='aruco_node',
            name='aruco_node',
            namespace='cam_right',
            parameters=[{
                'marker_size': 0.092,
                'aruco_dictionary_id': 'DICT_4X4_50',
                'image_topic': '/cam_right/image_raw',
                'camera_info_topic': '/cam_right/camera_info',
            }]
        ),

        # Aruco Node - RPI
        Node(
            package='ros2_aruco',
            executable='aruco_node',
            name='aruco_node',
            namespace='rpi',
            parameters=[{
                'marker_size': 0.092,
                'aruco_dictionary_id': 'DICT_4X4_50',
                'image_topic': '/rpi/image_raw',
                'camera_info_topic': '/rpi/camera_info',
            }]
        ),

    ])
