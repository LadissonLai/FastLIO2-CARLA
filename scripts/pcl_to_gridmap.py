#!/usr/bin/env python3

import time
import math
import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import tf2_ros


class PclToGridMap:

    def __init__(self):
        rospy.init_node('pcl_to_gridmap')

        self.resolution     = rospy.get_param('~resolution',      0.2)
        self.map_size       = rospy.get_param('~map_size',       800.0)
        self.max_range      = rospy.get_param('~max_range',       50.0)
        self.angle_step_deg = rospy.get_param('~angle_step_deg',   1.0)
        self.map_frame      = rospy.get_param('~map_frame',  'camera_init')
        self.robot_frame    = rospy.get_param('~robot_frame',    'body')

        self.lidar_height   = rospy.get_param('~lidar_height',     2.4)
        self.ground_tol     = rospy.get_param('~ground_tol',       0.3)
        self.obs_min_above  = rospy.get_param('~obs_min_above',    0.2)
        self.obs_max_above  = rospy.get_param('~obs_max_above',    3.0)

        self.kf_dist        = rospy.get_param('~keyframe_dist',    0.1)
        self.kf_angle       = rospy.get_param('~keyframe_angle',  3.0)

        self.n        = int(self.map_size / self.resolution)
        self.grid     = np.full((self.n, self.n), -1, dtype=np.int8)
        self.origin_x = None
        self.origin_y = None

        self.last_kf_x   = None
        self.last_kf_y   = None
        self.last_kf_yaw = None

        self._frame_total  = 0
        self._frame_update = 0

        self.tf_buf      = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        self.pub = rospy.Publisher('/local_map', OccupancyGrid, queue_size=1, latch=True)
        rospy.Subscriber(
            '/cloud_registered', PointCloud2, self.callback, queue_size=1)

        rospy.loginfo('[pcl2grid] Node started')
        rospy.loginfo(f'[pcl2grid] Map: {self.map_size}m x {self.map_size}m  '
                      f'resolution:{self.resolution}m  grid:{self.n}x{self.n}')
        rospy.loginfo(f'[pcl2grid] Keyframe threshold: '
                      f'translation>{self.kf_dist}m or rotation>{self.kf_angle}deg')

    def get_robot_pose(self):
        try:
            t = self.tf_buf.lookup_transform(
                self.map_frame, self.robot_frame,
                rospy.Time(0), rospy.Duration(0.05))
            tx  = t.transform.translation
            q   = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return tx.x, tx.y, tx.z, yaw
        except Exception as e:
            rospy.logwarn_throttle(3.0, f'[pcl2grid] TF failure: {e}')
            return None, None, None, None

    def is_keyframe(self, x, y, yaw):
        if self.last_kf_x is None:
            return True
        dist  = math.sqrt((x - self.last_kf_x)**2 + (y - self.last_kf_y)**2)
        d_yaw = abs(math.atan2(
            math.sin(yaw - self.last_kf_yaw),
            math.cos(yaw - self.last_kf_yaw)))
        return dist > self.kf_dist or d_yaw > math.radians(self.kf_angle)

    def in_bounds_mask(self, cx, cy):
        return (cx >= 0) & (cx < self.n) & (cy >= 0) & (cy < self.n)

    def callback(self, msg):
        self._frame_total += 1

        rx, ry, rz, yaw = self.get_robot_pose()
        if rx is None:
            return

        if not self.is_keyframe(rx, ry, yaw):
            return

        self.last_kf_x   = rx
        self.last_kf_y   = ry
        self.last_kf_yaw = yaw
        self._frame_update += 1

        t_start = time.perf_counter()

        ground_z = rz - self.lidar_height
        gnd_z_lo = ground_z - self.ground_tol
        gnd_z_hi = ground_z + self.ground_tol
        obs_z_lo = ground_z + self.obs_min_above
        obs_z_hi = ground_z + self.obs_max_above

        if self.origin_x is None:
            self.origin_x = rx - self.map_size / 2.0
            self.origin_y = ry - self.map_size / 2.0
            rospy.loginfo(f'[pcl2grid] Map origin: '
                          f'({self.origin_x:.1f}, {self.origin_y:.1f})  '
                          f'ground z={ground_z:.2f}m  '
                          f'obstacle z=[{obs_z_lo:.2f},{obs_z_hi:.2f}]m')

        t0  = time.perf_counter()
        raw = np.array(list(pc2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)),
            dtype=np.float32)
        t_parse = time.perf_counter() - t0

        if len(raw) == 0:
            return

        d2  = (raw[:, 0] - rx)**2 + (raw[:, 1] - ry)**2
        pts = raw[d2 < self.max_range**2]
        if len(pts) == 0:
            return

        t0 = time.perf_counter()

        dx       = pts[:, 0] - rx
        dy       = pts[:, 1] - ry
        angles   = np.arctan2(dy, dx)
        dists    = np.sqrt(dx**2 + dy**2)
        step_rad = np.deg2rad(self.angle_step_deg)
        n_bins   = int(2 * np.pi / step_rad)
        bin_idx  = ((angles + np.pi) / step_rad).astype(int) % n_bins

        bin_min_dist = np.full(n_bins, self.max_range, dtype=np.float32)
        np.minimum.at(bin_min_dist, bin_idx, dists)
        bin_angles = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)

        all_fcx, all_fcy = [], []
        for b in range(n_bins):
            d  = float(bin_min_dist[b])
            a  = float(bin_angles[b])
            fd = d * 0.90
            ns = max(2, int(fd / self.resolution))
            ts = np.linspace(0.0, fd, ns)
            xs = rx + ts * np.cos(a)
            ys = ry + ts * np.sin(a)
            cx = ((xs - self.origin_x) / self.resolution).astype(int)
            cy = ((ys - self.origin_y) / self.resolution).astype(int)
            ok = self.in_bounds_mask(cx, cy)
            all_fcx.append(cx[ok])
            all_fcy.append(cy[ok])

        if all_fcx:
            fcx = np.concatenate(all_fcx)
            fcy = np.concatenate(all_fcy)
            unk = self.grid[fcy, fcx] == -1
            self.grid[fcy[unk], fcx[unk]] = 0

        t_ray = time.perf_counter() - t0

        gnd = pts[(pts[:, 2] >= gnd_z_lo) & (pts[:, 2] <= gnd_z_hi)]
        if len(gnd) > 0:
            gcx = ((gnd[:, 0] - self.origin_x) / self.resolution).astype(int)
            gcy = ((gnd[:, 1] - self.origin_y) / self.resolution).astype(int)
            ok  = self.in_bounds_mask(gcx, gcy)
            unk = self.grid[gcy[ok], gcx[ok]] == -1
            self.grid[gcy[ok][unk], gcx[ok][unk]] = 0

        obs = pts[(pts[:, 2] >= obs_z_lo) & (pts[:, 2] <= obs_z_hi)]
        if len(obs) > 0:
            ocx = ((obs[:, 0] - self.origin_x) / self.resolution).astype(int)
            ocy = ((obs[:, 1] - self.origin_y) / self.resolution).astype(int)
            ok  = self.in_bounds_mask(ocx, ocy)
            self.grid[ocy[ok], ocx[ok]] = 100

        t0 = time.perf_counter()
        self._publish(msg.header.stamp)
        t_pub = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start
        # rospy.loginfo(
        #     f'[pcl2grid] KF#{self._frame_update} '
        #     f'(skipped {self._frame_total - self._frame_update} frames) | '
        #     f'total:{t_total*1000:.1f}ms '
        #     f'parse:{t_parse*1000:.1f}ms '
        #     f'ray:{t_ray*1000:.1f}ms '
        #     f'publish:{t_pub*1000:.1f}ms | '
        #     f'points:{len(pts)}')

    def _publish(self, stamp):
        out = OccupancyGrid()
        out.header.stamp              = stamp
        out.header.frame_id           = self.map_frame
        out.info.resolution           = self.resolution
        out.info.width                = self.n
        out.info.height               = self.n
        out.info.origin.position.x    = self.origin_x
        out.info.origin.position.y    = self.origin_y
        out.info.origin.position.z    = 0.0
        out.info.origin.orientation.w = 1.0
        out.data                      = self.grid.flatten().tolist()
        self.pub.publish(out)


if __name__ == '__main__':
    node = PclToGridMap()
    rospy.spin()