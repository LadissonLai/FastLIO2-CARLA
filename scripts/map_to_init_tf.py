#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
import tf2_ros
from geometry_msgs.msg import TransformStamped

published = False

def odom_cb(msg):
    global published
    if published:
        return

    t = TransformStamped()
    t.header.stamp    = rospy.Time.now()
    t.header.frame_id = "map"
    t.child_frame_id  = "camera_init"

    t.transform.translation.x = msg.pose.pose.position.x
    t.transform.translation.y = msg.pose.pose.position.y
    t.transform.translation.z = msg.pose.pose.position.z
    t.transform.rotation      = msg.pose.pose.orientation

    broadcaster.sendTransform(t)
    rospy.loginfo(f"[map_to_init] TF published: map -> camera_init "
                  f"({t.transform.translation.x:.2f}, "
                  f"{t.transform.translation.y:.2f}, "
                  f"{t.transform.translation.z:.2f})")
    published = True

if __name__ == '__main__':
    rospy.init_node('map_to_camera_init_tf')
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, odom_cb)
    rospy.spin()

