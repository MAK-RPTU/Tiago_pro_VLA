Setup the environment inside the container to avoid any issues:

python3 --version

cd /opt/pal/alum/share/ros2_ws

python3 -m venv venv

source venv/bin/activate

pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

place below packages in a requirements.txt file and run:

transformers
accelerate
huggingface_hub
safetensors
pillow
numpy
timm>=0.9.10,<1.0.0
tokenizers==0.22.2


pip install -r requirements.txt


hf auth whoami

hf auth login

ENter the Token you get from your hugging face login.

ENsure new python path:


echo $PYTHONPATH

export PYTHONPATH=/opt/pal/alum/share/hf_models/venv/lib/python3.10/site-packages:$PYTHONPATH 

Download Hugging face models:
huggingface-cli download openvla/openvla-v01-7b --local-dir openvla-7b

huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-spatial --local-dir openvla-7b-oft


Example to test VLA:

python run_openvla.py

Here is gazebo and moveit:

docker start <container>

docker stop <container>

docker exec -it <containerID> bash

If face write permissions to the container:
docker exec -it -u root <container_id> bash




**************************************
STARTUP
**************************************

FOr terminal issues double characters disable ibus:
pkill -9 ibus

sudo gedit /etc/xdg/autostart/im-launch.desktop

Add this line to the end of the file: Hidden=true

In local terminal:

If container is removed then run the image using below command:
docker run -it --name tiago_pro --network host --ipc=host --pid=host -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/workspace/tiago_pro/share:/opt/pal/alum/share:rw --gpus all --privileged --device=/dev/bus/usb --device=/dev/snd -w /opt/pal/alum/share development-tiago-pro-23:alum-25.01

IN local host
xhost +local:root

docker start tiago_pro

docker exec -it -u root tiago_pro bash

Run Gazebo without Rviz. Removed “actions.append(rviz_bringup_launch)” from tiago_pro_gazebo.launch.py

  (Broken package so avoid this start from next commands)  ros2 launch tiago_pro_gazebo tiago_pro_gazebo_nav_motion.launch.py use_sim_time:=True navigation:=True

COpy the table_woodenpeg from the git and put it into the pal_gazebo_worlds/worlds directory then run below:

ros2 launch tiago_pro_gazebo tiago_pro_gazebo.launch.py is_public_sim:=False use_sim_time:=True world_name:=table_woodenpeg

Run moveit_config

ros2 launch tiago_pro_moveit_config moveit_rviz.launch.py use_sim_time:=True

cd /opt/pal/alum/share/ros2_ws

source install/setup.bash


echo $PYTHONPATH

export PYTHONPATH=/opt/pal/alum/share/hf_models/venv/lib/python3.10/site-packages:$PYTHONPATH 

ros2 run vla_startup vla_arm_node 


Head controller:

ros2 topic pub --once /head_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "
joint_names:
- head_1_joint
- head_2_joint
points:
- positions: [0.2, -0.5]
  time_from_start:
    sec: 2
    nanosec: 0
"

Testing VLA to camera frame:
ros2 run tf2_tools view_frames

ros2 run tf2_ros tf2_echo base_link head_front_camera_color_optical_frame

ros2 topic echo /vla/camera_point

ros2 topic echo /vla/base_point

Verification: e.g
FRom tf: camera_z ≈ 1.183
From openvla: camera_point.z ≈ 0.002
ros2 run tf2_ros tf2_echo base_link head_front_camera_color_optical_frame
Translation: [0.070, 0.033, 1.183]
This tells you:
Camera is ~1.18 m above the base → ✔ correct for TIAGo

****************************
Moveit Node:
****************************

ros2 run vla_startup vla_frame_debug 


ros2 run tiago_moveit_vla vla_cartesian_executor \
  --ros-args -p arm:=left






*****************************************
FOr MCP server example turtlebot gazebo
~/workspace/tiago_pro/share/mcp_turtle
*****************************************

source /opt/ros/humble/setup.bash

export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

Then clone MCP server repo
cd ~/workspace/tiago_pro/share/mcp_turtle
git clone https://github.com/kakimochi/ros2-mcp-server.git
cd ros2-mcp-server

Follow remaining instructions over here:
https://github.com/kakimochi/ros2-mcp-server

And use the cline extension in vscode for MCP server configuration and to communicate with the ROS topics.

Please make the robot move forward at 0.2 m/s for 5 seconds.

It wil move the robot in gazebo once you approve in VScode



*****************************************
Roboneuron
/home/ubuntu/workspace/tiago_pro/share/RoboNeuron
*****************************************

