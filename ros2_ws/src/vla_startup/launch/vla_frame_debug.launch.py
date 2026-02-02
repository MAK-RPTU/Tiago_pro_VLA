#!/usr/bin/env python3
"""Launch vla_frame_debug with use_sim_time for TF / clock sync in simulation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time (required for TF and /clock in sim)",
    )

    node = Node(
        package="vla_startup",
        executable="vla_frame_debug",
        name="vla_frame_debug",
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    return LaunchDescription([use_sim_time_arg, node])
