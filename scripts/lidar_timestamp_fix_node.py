#!/usr/bin/env python3

import rospy
import numpy as np
import struct
from sensor_msgs.msg import PointCloud2, PointField

NUM_CHANNELS      = rospy.get_param('/lidar_fix/num_channels',  32)
UPPER_FOV_DEG     = rospy.get_param('/lidar_fix/upper_fov',     5.0)
LOWER_FOV_DEG     = rospy.get_param('/lidar_fix/lower_fov',    -25.0)
ROTATION_HZ       = rospy.get_param('/lidar_fix/rotation_hz',   10.0)

SCAN_PERIOD   = 1.0 / ROTATION_HZ
FOV_RANGE_DEG = UPPER_FOV_DEG - LOWER_FOV_DEG
DEG_PER_RING  = FOV_RANGE_DEG / (NUM_CHANNELS - 1)

def fix_cloud(msg):
    n   = msg.width * msg.height
    if n == 0:
        return

    src_step = msg.point_step
    src_data = bytearray(msg.data)

    fmap = {f.name: f.offset for f in msg.fields}
    if 'x' not in fmap:
        return

    ox, oy, oz = fmap['x'], fmap['y'], fmap['z']

    new_step = src_step + 2 + 4
    off_ring = src_step
    off_time = src_step + 2

    new_fields = list(msg.fields) + [
        PointField('ring', off_ring, PointField.UINT16,  1),
        PointField('time', off_time, PointField.FLOAT32, 1),
    ]

    new_data = bytearray(n * new_step)

    for i in range(n):
        ss = i * src_step
        ds = i * new_step

        new_data[ds: ds + src_step] = src_data[ss: ss + src_step]

        x = struct.unpack_from('f', src_data, ss + ox)[0]
        y = struct.unpack_from('f', src_data, ss + oy)[0]
        z = struct.unpack_from('f', src_data, ss + oz)[0]

        r_xy = np.sqrt(x*x + y*y)
        elev_deg = np.degrees(np.arctan2(z, r_xy)) if r_xy > 0.001 else 0.0
        ring_idx = int(round((elev_deg - LOWER_FOV_DEG) / DEG_PER_RING))
        ring_idx = max(0, min(NUM_CHANNELS - 1, ring_idx))

        azimuth  = np.arctan2(y, x)
        ratio    = (azimuth + np.pi) / (2.0 * np.pi)
        time_val = float(ratio * SCAN_PERIOD)

        struct.pack_into('H', new_data, ds + off_ring, ring_idx)
        struct.pack_into('f', new_data, ds + off_time, time_val)

    out             = PointCloud2()
    out.header      = msg.header
    out.height      = msg.height
    out.width       = msg.width
    out.fields      = new_fields
    out.is_bigendian = msg.is_bigendian
    out.point_step  = new_step
    out.row_step    = new_step * msg.width
    out.data        = bytes(new_data)
    out.is_dense    = msg.is_dense
    pub.publish(out)

if __name__ == '__main__':
    rospy.init_node('lidar_timestamp_fix')
    INPUT  = rospy.get_param('~input_topic',  '/carla/ego_vehicle/lidar')
    OUTPUT = rospy.get_param('~output_topic', '/points_raw')
    pub = rospy.Publisher(OUTPUT, PointCloud2, queue_size=10)
    rospy.Subscriber(INPUT, PointCloud2, fix_cloud, queue_size=10)
    rospy.loginfo(f"[lidar_fix] {INPUT} -> {OUTPUT}  "
                f"channels={NUM_CHANNELS}  fov=[{LOWER_FOV_DEG},{UPPER_FOV_DEG}]deg")
    rospy.spin()