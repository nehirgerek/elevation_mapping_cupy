"""Launch elevation_mapping_cupy fed by DLIO's deskewed point cloud, with a
`quad` argument for multi-robot namespacing (matching the convention already
established across this workspace's acl-mapping/travel/DLIO launch files).

Usage (root namespace / single robot / bag playback):
  ros2 launch elevation_mapping_cupy elevation_mapping_dlio.launch.py

Usage (namespaced hardware run):
  ros2 launch elevation_mapping_cupy elevation_mapping_dlio.launch.py quad:=RR08

core_param.yaml and config/setups/dlio/base.yaml are loaded here as
flattened Python dicts (not raw file paths) rather than via `parameters=[file_path]`
directly, because core_param.yaml uses ROS2's absolute-node-name YAML key
(`/elevation_mapping_node:`), which ROS2 only applies to a node whose fully
qualified name matches EXACTLY -- it silently no-ops once this node is
namespaced (e.g. `/RR08/elevation_mapping_node`). A plain dict passed via
`Node(parameters=[...])` always applies regardless of namespace. This is the
same workaround already used in acl-mapping's global_mapper_travel.launch.py
for travel_ros's equally absolute-keyed kitti_params.yaml -- see that file's
comment for the original discovery (GroundGrid hit the identical issue with
mid360.yaml).
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _load_ros_parameters(yaml_path: str) -> dict:
    """Extract the ros__parameters dict regardless of whether the file uses
    the `/**:` wildcard key or an absolute `/<node_name>:` key -- both are
    valid ROS2 param-file top-level keys, and this launch file bypasses
    ROS2's own node-name matching entirely by flattening to a plain dict
    (see this file's header comment), so either convention works here."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    for key, value in data.items():
        if isinstance(value, dict) and 'ros__parameters' in value:
            return value['ros__parameters']
    raise ValueError(f"{yaml_path}: no '<key>: {{ros__parameters: ...}}' section found")


def launch_setup(context, *args, **kwargs):
    quad = LaunchConfiguration('quad').perform(context)
    deskewed_cloud_topic = LaunchConfiguration('deskewed_cloud_topic').perform(context)
    map_frame_id = LaunchConfiguration('map_frame_id').perform(context)
    base_frame_id = LaunchConfiguration('base_frame_id').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    launch_rviz = LaunchConfiguration('launch_rviz')
    rviz_config = LaunchConfiguration('rviz_config').perform(context)

    share_dir = get_package_share_directory('elevation_mapping_cupy')
    core_param_path = os.path.join(share_dir, 'config', 'core', 'core_param.yaml')
    dlio_param_path = os.path.join(share_dir, 'config', 'setups', 'dlio', 'base.yaml')
    traversability_param_path = os.path.join(
        share_dir, 'config', 'core', 'traversability_plugin_config.yaml')

    params = {}
    params.update(_load_ros_parameters(core_param_path))
    params.update(_load_ros_parameters(dlio_param_path))
    # Geometric traversability pipeline (normals/slope/roughness/step/fusion/occupancy/ESDF),
    # loaded as a second plugin file merged on top of plugin_config.yaml -- see
    # parameter.py:extra_plugin_config_file and elevation_mapping.py's load_plugin_settings
    # call. Each stage still has its own enable:true/false in that file; this just makes the
    # file itself available to load.
    params['extra_plugin_config_file'] = traversability_param_path

    # Overrides -- see dlio/base.yaml's header comment for why these three
    # (topic/map_frame/base_frame) are placeholders in the file itself.
    params['subscribers'] = {
        'dlio_cloud': {
            'topic_name': deskewed_cloud_topic,
            'data_type': 'pointcloud',
        }
    }
    params['map_frame'] = map_frame_id
    params['base_frame'] = base_frame_id
    params['corrected_map_frame'] = map_frame_id
    params['use_sim_time'] = use_sim_time.lower() in ('true', '1')

    elevation_mapping_node = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node.py',
        name='elevation_mapping_node',
        namespace=quad if quad else None,
        output='screen',
        parameters=[params],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        namespace=quad if quad else None,
        arguments=['-d', rviz_config] if rviz_config else [],
        parameters=[{'use_sim_time': use_sim_time.lower() in ('true', '1')}],
        output='screen',
        condition=IfCondition(launch_rviz),
    )

    return [elevation_mapping_node, rviz_node]


def generate_launch_description() -> LaunchDescription:
    # Mirrors dlio/odom.cc's own frame-namespacing (get_namespace(), leading
    # slash stripped, "<ns>/odom" / "<ns>/base_link") and this workspace's
    # other DLIO-consumer launch files (dlio_deskewed_elevation.launch.py) --
    # empty quad = root namespace, matching DLIO's own "skip prefixing
    # entirely at the root namespace" behavior.
    quad_arg = DeclareLaunchArgument(
        'quad', default_value='',
        description='Robot namespace DLIO is launched under, e.g. RR08. '
                    'Empty (default) = root namespace / bag playback.',
    )

    quad = LaunchConfiguration('quad')
    default_deskewed_topic = PythonExpression(
        ["'/' + '", quad, "' + '/dlio/odom_node/pointcloud/deskewed' if '", quad, "' else 'dlio/odom_node/pointcloud/deskewed'"])
    default_map_frame_id = PythonExpression(["'", quad, "/odom' if '", quad, "' else 'odom'"])
    default_base_frame_id = PythonExpression(["'", quad, "/base_link' if '", quad, "' else 'base_link'"])

    deskewed_topic_arg = DeclareLaunchArgument(
        'deskewed_cloud_topic', default_value=default_deskewed_topic,
        description='DLIO deskewed PointCloud2 topic (dlio/odom.cc: deskewed_pub).',
    )
    map_frame_arg = DeclareLaunchArgument(
        'map_frame_id', default_value=default_map_frame_id,
        description="Must match DLIO's this->odom_frame exactly.",
    )
    base_frame_arg = DeclareLaunchArgument(
        'base_frame_id', default_value=default_base_frame_id,
        description="Must match DLIO's this->baselink_frame exactly.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation clock if true (e.g. bag playback with --clock).',
    )
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz', default_value='false', description='Whether to launch RViz.',
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value='', description='Path to an RViz config file.',
    )

    return LaunchDescription([
        quad_arg,
        deskewed_topic_arg,
        map_frame_arg,
        base_frame_arg,
        use_sim_time_arg,
        launch_rviz_arg,
        rviz_config_arg,
        OpaqueFunction(function=launch_setup),
    ])
