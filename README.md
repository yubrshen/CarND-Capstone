
This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: 
Programming a Real Self-Driving Car. 

The project is successfully delivered by the team of "Cruising-by-the-Bay" of Udacity Self-Driving-Car students. 

Here are the team members:


Name                   | email                    | Role
-----------------------|--------------------------|-------
Yu Shen                | yubrshen@gmail.com       | team lead
Peng Zhang             | zpactfc@gmail.com        | 
Sumanth Reddy Kaliki   | sumanth818@gmail.com     |
Sahil Juneja           | sahiljuneja17@gmail.com  |
Hector Sanchez Pajares | hector.spc@gmail.com     |

# LIMITATION

When simulator is started and "Manual" unchecked before the ROS nodes are started and let to finish initialization. 
The message of "dbw_enabled" might be lost. Then the car would not receive /vehicle/throttle_cmd, etc. 
The car would not move along the track properly. 

To overcome the potential problem, please start ROS nodes and let them finish initialization, then start the simulator, and then uncheck "Manual". 
Likewise, to drive the real car, the ROS nodes should also be started and let to initialize first then engage dbw to drive. 

This ends the limitation. 

# Project Descriptions 

The project bases on the architecture proposed by Udacity. It uses ROS as the implementation framework. 
The project implements the following ROS nodes:

- waypoint_updater
- tl_detector
- dbw_node

It also implements a deep learning classifier to classify traffic light's colors. By learning from examples, the classifier can classify 
traffic light samples in both the simulated environment, and real traffic light images. Experiements shows that the classification can provide 
sufficient level of correctness for traffic light color classification. 

Extensive experiments have shown that the system can drive a car autonomously correctly on the simulated track
following the traffic light signals. Based the requirements, the car should stop at the end of the 
track. The system is designed, built to drive real car. Tests will be conducted to drive real car in a parking lot of Udacity. 

Besides adhering to the established architecture, special care is made to pre-compute and store required distance computation, 
so that the real-time computation is minimized, which may improve the responsiveness, and reliability of the Self-Driving-Car system.

For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

To review the design artifacts (design document and source code), goto [Design in Literate Programming](./capstone-workbook.org)

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
