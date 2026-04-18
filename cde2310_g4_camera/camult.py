from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Static TF publishers  (camera frames → base_link)
    #    Format: [x, y, z, yaw, pitch, roll, parent, child]
    #    NOTE: cam_left / cam_right values below are placeholders — update them.
    # ──────────────────────────────────────────────────────────────────────────
    camera_transforms = [
        [ 0.10391, 0.0, 0.02824, 0, 0, 0, 'base_link', 'cam_left_camera_link'],   # TODO: verify values
        [ 0.0805,  0.0, 0.11015, 0, 0, 0, 'base_link', 'cam_right_camera_link'],  # TODO: verify values
        # Add rpi_camera_link → base_link here when measurements are ready:
        # [ x, y, z, yaw, pitch, roll, 'base_link', 'rpi_camera_link'],
    ]

    nodes = []

    for x, y, z, yaw, pitch, roll, parent, child in camera_transforms:
        nodes.append(
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'static_tf_{child}',
                arguments=[
                    '--x',           str(x),
                    '--y',           str(y),
                    '--z',           str(z),
                    '--roll',        str(roll),
                    '--pitch',       str(pitch),
                    '--yaw',         str(yaw),
                    '--frame-id',    parent,
                    '--child-frame-id', child,
                ],
            )
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 2. USB camera nodes  (usb_cam package)
    #    Toggle which cameras to launch by editing the list below.
    #    Each entry: (node_name, device, namespace, camera_info_url)
    # ──────────────────────────────────────────────────────────────────────────
    usb_cameras = [
        (
            'usb_cam_left',
            '/dev/video0',
            'cam_left',
            'file:///home/ubuntu/.ros/camera_info/640480left.yaml',
        )
     #   (
    #        'usb_cam_right',
   #         '/dev/video2',
  #          'cam_right',
 #           'file:///home/ubuntu/.ros/camera_info/640480right.yaml',
#        ),
    ]

    for name, device, ns, info_url in usb_cameras:
        nodes.append(
            Node(
                package='usb_cam',
                executable='usb_cam_node_exe',
                name=name,
                namespace=ns,
                parameters=[{
                    'video_device':    device,
                    'image_width':     640,
                    'image_height':    480,
                    'pixel_format':    'mjpeg2rgb',
                    'frame_id':        f'{ns}_camera_link',
                    'framerate':       30.0,
                    'camera_info_url': info_url,
                }],
            )
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. RPI camera node  (v4l2_camera package) — uncomment to enable
    # ──────────────────────────────────────────────────────────────────────────
    """
    nodes.append(
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='rpi',
            namespace='rpi',
            parameters=[{
                'video_device':    '/dev/video4',
                'image_size':      [320, 240],
                'camera_frame_id': 'rpi_camera_link',
                'frame_id':        'rpi_camera_link',
                'pixel_format':    'YUYV',
                'image_encoding':  'yuv422_yuy2',
                'fps':             10.0,
                'camera_info_url': 'file:///home/ubuntu/.ros/camera_info/rpi.yaml',
            }],
        )
    )
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 4. ArUco detector nodes  (ros2_aruco package)
    # ──────────────────────────────────────────────────────────────────────────
    aruco_cameras = [
        ('cam_left',  '/cam_left/image_raw',  '/cam_left/camera_info'),
        ('cam_right', '/cam_right/image_raw', '/cam_right/camera_info'),
        # Uncomment when rpi camera is enabled:
        # ('rpi',     '/rpi/image_raw',       '/rpi/camera_info'),
    ]

    for ns, image_topic, info_topic in aruco_cameras:
        nodes.append(
            Node(
                package='ros2_aruco',
                executable='aruco_node',
                name='aruco_node',
                namespace=ns,
                parameters=[{
                    'marker_size':          0.092,
                    'aruco_dictionary_id':  'DICT_4X4_50',
                    'image_topic':          image_topic,
                    'camera_info_topic':    info_topic,
                }],
            )
        )

    return LaunchDescription(nodes)
