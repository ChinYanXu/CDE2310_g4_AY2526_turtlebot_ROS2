from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Define USB Cameras (using usb_cam package)
    # ('Name', 'Device Path', 'Namespace')
    usb_cameras = [
        ('usb_cam_right', '/dev/video0', 'cam_right','file:///home/ubuntu/.ros/camera_info/cam_right.yaml'),
        ('usb_cam_left', '/dev/video2', 'cam_left','file:///home/ubuntu/.ros/camera_info/cam_left.yaml')
    ]

    # 2. Define RPI Camera (using v4l2_camera package)
    rpi_cam_config = {'name': 'rpi', 'device': '/dev/video4', 'ns': 'rpi'}

    nodes = []

    # Add USB Camera Nodes
    for name, device, ns,camerainfourl in usb_cameras:
        nodes.append(
            Node(
                package='usb_cam',
                executable='usb_cam_node_exe',
                name=name,
                namespace=ns,
                parameters=[{
                    'video_device': device,
                    'image_width': 320,
                    'image_height': 240,
                    'pixel_format': 'mjpeg2rgb',
                    'frame_id': f'{ns}_camera_link',
                    'fps': 20.0,  # Keep FPS low for LiDAR/OpenCR stability
		    'camera_info_url' : camerainfourl
                }],
#		remappings=[
 #               	('image_raw', 'image_raw_unused'),
#            	]
            )
        )

    # Add RPI Camera Node
    nodes.append(
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name=rpi_cam_config['name'],
            namespace=rpi_cam_config['ns'],
            parameters=[{
                'video_device': rpi_cam_config['device'],
                'image_size': [640,480],
                'camera_frame_id': f"{rpi_cam_config['ns']}_camera_link",
		'frame_id' : f"{rpi_cam_config['ns']}_camera_link",
		'fps': 30.0,
		'camera_info_url' : 'file:///home/ubuntu/.ros/camera_info/rpi.yaml',
            }]
        )
    )

    return LaunchDescription(nodes)