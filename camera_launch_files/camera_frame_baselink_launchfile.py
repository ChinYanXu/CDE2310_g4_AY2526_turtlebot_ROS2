from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Define your transforms here for easy editing
    # Format: [x, y, z, yaw, pitch, roll, parent, child]
    camera_transforms = [
        # [-0.00295,  0.09795, 0.11795,  1.5707963, 0, 0, 'base_link', 'cam_left_camera_link'],
        #'x_l', 'y_l', 'z_l', 'yaw_l', 'pitch_l', 'roll_l', THESE VALUES ARE WRONG plz change
        # [-0.00323, -0.09785, 0.1178,  -1.5707963, 0, 0, 'base_link', 'cam_right_camera_link'],
        [ 0.10391, 0.0, 0.02824,  0, 0, 0, 'base_link', 'cam_left_camera_link'], #bottom
        #'x_r', 'y_r', 'z_r', 'yaw_r', 'pitch_r', 'roll_r',
        [ 0.0805, 0.0,  0.11015,   0,      0, 0, 'base_link', 'cam_right_camera_link'], #top
        #'x_rp', 'y_rp', 'z_rp', 'yaw_rp', 'pitch_rp', 'roll_rp',
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

    return LaunchDescription(nodes)
