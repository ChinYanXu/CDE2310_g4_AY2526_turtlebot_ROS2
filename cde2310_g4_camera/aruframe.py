from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
																																																																																																																																		
    # ── Static TF Publishers (base_link → camera frames) ──────────────────────
    # Format: [x, y, z, yaw, pitch, roll, parent, child]
    camera_transforms = [
        [ 0.10391, 0.0, 0.02824,  0, 0, 0, 'base_link', 'cam_left_camera_link'],
        #'x_l', 'y_l', 'z_l', 'yaw_l', 'pitch_l', 'roll_l', THESE VALUES ARE WRONG plz change
        [ 0.0805, 0.0,  0.11015,  0, 0, 0, 'base_link', 'cam_right_camera_link'], 
        #'x_r', 'y_r', 'z_r', 'yaw_r', 'pitch_r', 'roll_r',
    ]

    nodes = []
    for x,y,z,yaw,pitch,roll,parent,child in camera_transforms:
        nodes.append(
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'static_tf_publisher_{child}',
                arguments= [
                    '--x', str(x),
                    '--y', str(y),
                    '--z', str(z),
                    '--roll', str(roll),
                    '--pitch', str(pitch),
                    '--yaw', str(yaw),
                    '--frame-id', parent,
                    '--child-frame-id', child
                ]
            )
        )

    # ── ArUco Detection Nodes ──────────────────────────────────────────────────
    aruco_configs = [
        ('cam_left',  '/cam_left/image_raw/compressed',  '/cam_left/camera_info'),
        ('cam_right', '/cam_right/image_raw/compressed', '/cam_right/camera_info'),
        ('rpi',       '/rpi/image_raw/compressed',       '/rpi/camera_info'),
    ]

    for namespace, image_topic, camera_info_topic in aruco_configs:
        nodes.append(
            Node(
                package='ros2_aruco',
                executable='aruco_node',
                name='aruco_node',
                namespace=namespace,
                parameters=[{
                    'marker_size': 0.093,
                    'aruco_dictionary_id': 'DICT_4X4_50',
                    'image_topic': image_topic,
                    'camera_info_topic': camera_info_topic,
                }]
            )
        )

    return LaunchDescription(nodes)
