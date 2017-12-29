#!/usr/bin/env python
import sys                      # for redirect stderr
import rospy

import copy                     # for deepcopy
import numpy as np              # for polyfit and poly1d

import math

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from styx_msgs.msg import TrafficLightArray
from waypoint_lib.waypoint_tracker import WaypointTracker

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
LOOKAHEAD_WPS = 30 # 200 # Number of waypoints we will publish. You can change this number

MAX_DECEL = 3.0 # 5.0 reduce to 3.0 to model the fact that the deceleration is rather slow
MAX_ACCEL = 1.0
# SAFE_DIST = 5 # 27.0 # 25 is good value to stop, but too far from the light,
# 17 is better than 25 before the change of filter only do when non_red_to_red

KMH_to_MPS = 1000.0/3600.0   # 1 Kilo-Meters = 1000 meters, 1 hour = 3600 seconds

def publish_Lane(publisher, waypoints):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        publisher.publish(lane)
def noneORfloat(y):
    return y if y is None else float(y)

def log_update_state(car_index, light_index_or_last, if_RED, dist_to_light, min_stop_dist, current_velocity, comment):
    label = ("car index {:4} " +
             "light index {:4} " +
             "curr. light color: {:7} " +
             "dist. to light: {:7.2} " +
             "min. stop dist. {:7.2} " +
             "curr. vel. {:7.2}; " +
             comment)
    rospy.loginfo(label.format(car_index,
                               light_index_or_last,
                               if_RED,
                               noneORfloat(dist_to_light),
                               noneORfloat(min_stop_dist),
                               noneORfloat(current_velocity)))

class WaypointUpdater(WaypointTracker):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        self.max_vel_mps = rospy.get_param('waypoint_loader/velocity')*KMH_to_MPS
        # MPH_to_MPS, confirmed that the unit is KMH
        self.SAFE_DIST = rospy.get_param('~SAFE_DIST', 5)

        rospy.loginfo('max_vel_mps: %f' % self.max_vel_mps)
        self.loop_freq = rospy.get_param('~loop_freq', 2)
        # the frequency to process vehicle messages

        WaypointTracker.__init__(self)

        self.current_pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb)

        # DONE: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.traffic_waypoint = None
        self.new_traffic_waypoint = False  # whether there is new traffic_waypoint data to process
        self.traffic_light_red = False
        self.prev_traffic_light_red = False

        self.obstacle_waypoint = None
        self.current_velocity = None
        self.velocity_policy = None

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_waypoint_cb)

        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(2)                # in initialization loop less frequent
        while not rospy.is_shutdown():
            rate = rospy.Rate(self.loop_freq)
            if self.ready and self.pose:
                # rospy.loginfo(("the number of elements in self.base_waypoints: {}"+
                #               " before accessing in get car index").format(len(self.base_waypoints)))
                tmp = self.get_closest_waypoint(self.pose.pose)
                self.car_index, local_x, local_y = tmp if tmp else (None, None, None)
                # as side effect stored in self.car_index
                
                if self.car_index is not None:
                    _, self.traffic_waypoint = self.waypoint_to_light[self.car_index]
                    # compute minimum_stop_dist to consider if need braking
                
                    if self.current_velocity is not None:
                        min_stop_dist = self.current_velocity**2 / (2.0 * MAX_DECEL) + self.SAFE_DIST
                    else:
                        min_stop_dist = self.SAFE_DIST
                    # end of if self.current_velocity is not None
                
                    light_index_or_last = (self.traffic_waypoint if self.traffic_waypoint is not None
                                           else len(self.base_waypoints)-1)
                
                    tl_dist = (self.distance(self.car_index, light_index_or_last))
                
                    def assemble_final_waypoints():  # for the case where only need to have the first element deepcopy'ed
                        return (
                            [copy.deepcopy(self.base_waypoints[self.car_index])] +
                            self.base_waypoints[self.car_index+1: (self.car_index + LOOKAHEAD_WPS)]
                            if self.car_index < len(self.base_waypoints)-1 else [])
                
                    if ((self.car_index <= light_index_or_last) and
                        (self.traffic_light_red or (light_index_or_last == (len(self.base_waypoints)-1)))):
                        if (tl_dist < min_stop_dist):
                            if (self.car_index <= light_index_or_last):
                                final_waypoints = []
                                for i in range(self.car_index, light_index_or_last+1):
                                    final_waypoints.append(copy.deepcopy(self.base_waypoints[i]))
                                # end of for i in range(self.car_index, self.traffic_waypoint)
                                final_waypoints = self.decelerate(self.car_index, light_index_or_last, final_waypoints)
                            # end of if (self.car_index <= light_index_or_last)
                            # log_update_state(car_index=self.car_index,
                            #                  light_index_or_last=light_index_or_last,
                            #                  if_RED="RED" if self.traffic_light_red else "not-RED",
                            #                  dist_to_light=tl_dist,
                            #                  min_stop_dist=min_stop_dist,
                            #                  current_velocity=self.current_velocity,
                            #                  comment="within stop dist., decelerate")
                        else:                   # too far to brake
                            final_waypoints = assemble_final_waypoints()
                            # log_update_state(car_index=self.car_index,
                            #                  light_index_or_last=light_index_or_last,
                            #                  if_RED="RED" if self.traffic_light_red else "not-RED",
                            #                  dist_to_light=tl_dist,
                            #                  min_stop_dist=min_stop_dist,
                            #                  current_velocity=self.current_velocity,
                            #                  comment="too far to brake, no slow down")
                        # end of if (tl_dist < min_stop_dist)
                    else:                       # no traffic light ahead or no turning red light
                        final_waypoints = assemble_final_waypoints()
                        # log_update_state(car_index=self.car_index,
                        #                  light_index_or_last=light_index_or_last,
                        #                  if_RED="RED" if self.traffic_light_red else "not-RED",
                        #                  dist_to_light=tl_dist,
                        #                  min_stop_dist=min_stop_dist,
                        #                  current_velocity=self.current_velocity,
                        #                  comment="no red traffic light ahead, keep the curr. vel.")
                    # end of ((self.traffic_waypoint is not None) and
                    # (self.car_index <= self.traffic_waypoint) and
                    # (self.traffic_light_red or (light_index_or_last == (len(self.base_waypoints)-1))))
                
                    # adjust the angular velocity for final_waypoints[0]
                    # in order to return to the track when the current pose is off track
                    final_waypoints[0].twist.twist.angular.z = math.atan2(local_y, local_x)
                    #turning_coff = 1.0
                    # final_waypoints[0].twist.twist.linear.x = ( # reduce the speed proportional to the offset angle
                    #     final_waypoints[0].twist.twist.linear.x*(1.0/(1+turning_coff*abs(final_waypoints[0].twist.twist.angular.z))))
                
                    # publish to /final_waypoints, need to package final_waypoints into Lane message
                    publish_Lane(self.final_waypoints_pub, final_waypoints)
                # end of if self.car_index is not None
                
                self.pose = None        # indicating this message has been processed
            # end of if self.ready and self.pose
            rate.sleep()
        # end of while not rospy.is_shutdow()

    def preprocess(self):
        if self.base_waypoints:
            WaypointTracker.preprocess(self)
            rospy.loginfo(("the number of elements in self.base_waypoints: {}"+
                           " at the exit of base_waypoint_cb").format(len(self.base_waypoints)))
            self.ready = True
            # end of if self.base_waypoints

    def decelerate(self, start, end, waypoints):
        """
        arrange the velocities of the waypoints such that
        waypoints[-1].linear.x = 0
        and the deceleration should be smooth.
        waypoints are an array of waypoints to have velocity reduced.
        start and end are the index in the self.base_waypoints array
        for the start and the end of the waypoints.
        """
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.0
        for i in range(len(waypoints)-2, -1, -1):
            wp = waypoints[i]
            dist = self.distance(i+start, end)
            dist = max(0.0, dist-self.SAFE_DIST)
            vel  = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            rospy.loginfo("wp.twist.twist.linear.x {}".format(wp.twist.twist.linear.x))
        return waypoints
    

    def base_waypoints_cb(self, msg):
        WaypointTracker.base_waypoints_process(self, msg)
    

    def current_pose_cb(self, msg):
        # WORKING: Implement
        #
        if self.pose is None:       # ready to process message
            self.pose = msg
        # end of if self.pose is None
        # otherwise, the current message is being processed, rejected the coming message and expect to receive more updated next one.

    def traffic_waypoint_cb(self, msg):
        existing_traffic_waypoint = (self.traffic_waypoint * (1 if self.traffic_light_red else -1)
                                     if self.traffic_waypoint is not None else 0)
        if existing_traffic_waypoint != msg.data:
            self.new_traffic_waypoint = True
            self.prev_traffic_light_red = self.traffic_light_red
            self.traffic_light_red = (0 <= msg.data)
            self.traffic_waypoint = abs(msg.data)
        else:
            self.new_traffic_waypoint = False
        # end of if self.traffic_waypoint != msg.data
        rospy.loginfo("self.traffic_light_red: {} msg.data {} in traffic_waypoint_cb".format(self.traffic_light_red, msg.data))
    
    def red_to_non_red(self):
        return (self.new_traffic_waypoint and
                self.prev_traffic_light_red and
                (not self.traffic_light_red))
    
    def non_red_to_red(self):
        return (self.new_traffic_waypoint and
                (not self.prev_traffic_light_red) and
                self.traffic_light_red)

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def obstacle_cb(self, msg):
        self.obstacle_waypoint = msg.data

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
