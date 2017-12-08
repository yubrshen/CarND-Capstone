#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
import tf

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
LOOP_RATE = 2
DISTANCE_TO_STOP = 6
DISTANCE_TO_DECEL = 25

class CarState:
    ACCEL = 0
    DECEL = 1
    STOP = 2
    KEEP = 3

class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        # DONE: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.current_pose = None
        self.base_waypoints = None
        self.tl_waypoint = None
        self.obstable_waypoint = None
        self.final_waypoints = None
        self.current_velocity = None
        self.tl_is_red = False
        self.tl_changed = False
        self.car_state = CarState.STOP

        # rospy.spin()

        rate = rospy.Rate(LOOP_RATE)
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

    def loop(self):
        if self.base_waypoints is None or self.current_pose is None:
            return

        next_index = self.get_next_waypoint(self.current_pose)
        if next_index is None:
            return

        self.final_waypoints = self.base_waypoints[next_index:(next_index + LOOKAHEAD_WPS)]
        if self.tl_waypoint is not None and self.tl_waypoint >= next_index:
            distance_to_tl = self.distance(self.base_waypoints, next_index, self.tl_waypoint)
            rospy.loginfo("distant to TL {}".format(distance_to_tl))
            if self.tl_changed and not self.tl_is_red:
                self.car_state = CarState.ACCEL
                rospy.loginfo("TL CHanged 2 GREEN")
            elif self.tl_changed and self.tl_is_red:
                self.car_state = CarState.DECEL
                rospy.loginfo("TL CHanged 2 RED")
            elif self.tl_is_red and distance_to_tl <= DISTANCE_TO_STOP:
                self.car_state = CarState.STOP
                rospy.loginfo("STOP NOW")
            elif self.tl_is_red and distance_to_tl < DISTANCE_TO_DECEL:
                self.car_state = CarState.DECEL
                rospy.loginfo("We decelerate ")
            else:
                self.car_state = CarState.ACCEL
                rospy.loginfo("We accelerate ")

            for i in range(len(self.final_waypoints)):
                cruise_speed = self.get_waypoint_velocity(self.final_waypoints[i])
                tl_index = self.tl_waypoint - next_index
                if self.car_state is CarState.STOP:
                    if i < tl_index:
                        target_speed = -1.0
                    else:
                        target_speed = cruise_speed
                elif self.car_state is CarState.ACCEL:
                    target_speed = cruise_speed
                elif self.car_state is CarState.DECEL:
                    distance_to_tl = self.distance(self.final_waypoints, i, tl_index)
                    if DISTANCE_TO_STOP < distance_to_tl:
                        target_speed = -1.0
                    else:
                        accel = 1.0
                        distance_to_brake = DISTANCE_TO_STOP - distance_to_tl
                        target_speed = max(math.sqrt(2 * accel * distance_to_brake), 0)
                else:
                    target_speed = self.current_velocity

                target_speed = min(cruise_speed, target_speed)
                rospy.loginfo("target speed {} / cruise vel {}".format(target_speed, cruise_speed))
                self.set_waypoint_velocity(self.final_waypoints[i], target_speed)
                rospy.loginfo("wp speed {}".format(self.final_waypoints[i].twist.twist.linear.x))
        else:
            self.tl_waypoint = None


        # rospy.loginfo("next index {}".format(next_index))

        self.publish_final_waypoints()


    def publish_final_waypoints(self):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = self.final_waypoints

        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        self.current_pose = msg.pose

    def waypoints_cb(self, lane):
        if self.base_waypoints is None:
            self.base_waypoints = lane.waypoints

    def traffic_cb(self, msg):
        if self.tl_waypoint != msg.data:
            wp = abs(msg.data)
            if self.tl_waypoint is wp:
                self.tl_changed = True
            else:
                self.tl_waypoint = wp
            self.tl_is_red = msg.data >= 0

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        self.obstable_waypoint = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += self.dist(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    # Path planning project
    # https://github.com/hecspc/CarND-Path-Planning-Project/blob/master/src/main.cpp

    def dist(self, p1, p2):
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2)

    def get_closest_waypoint(self, pose):

        closest_len = 100000
        closest_index = 0

        waypoints = self.base_waypoints

        for i in range(len(waypoints)):
            waypoint = waypoints[i].pose.pose.position
            d = self.dist(pose.position, waypoint)
            if d < closest_len:
                closest_len = d
                closest_index = i

        return closest_index

    def get_next_waypoint(self, pose):

        next_index = self.get_closest_waypoint(pose)
        p1 = pose.position
        p2 = self.base_waypoints[next_index].pose.pose.position

        heading = math.atan2((p2.y - p1.y), (p2.x - p1.x))

        quaternion = (pose.orientation.x,
                      pose.orientation.y,
                      pose.orientation.z,
                      pose.orientation.w)

        # https://answers.ros.org/question/69754/quaternion-transformations-in-python/
        euler_orientation = tf.transformations.euler_from_quaternion(quaternion)
        # roll = euler_orientation[0]
        # pitch = euler_orientation[1]
        yaw = euler_orientation[2]

        angle = abs(yaw - heading)

        if angle > math.pi / 4:
            next_index += 1

        return next_index


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
