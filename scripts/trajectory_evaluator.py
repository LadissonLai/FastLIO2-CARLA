#!/usr/bin/env python3
"""
Trajectory Evaluation Script
Subscribes to CARLA ground truth and FAST-LIO2 trajectory, generates comparison plot and performance report upon Ctrl+C.

Subscriptions:
  /carla/ego_vehicle/odometry  (nav_msgs/Odometry) -- CARLA Ground Truth
  /Odometry_fastlio            (nav_msgs/Odometry) -- FAST-LIO2 Estimation

Outputs:
  trajectory_compare.png  -- Trajectory comparison plot
  trajectory_report.txt   -- Performance report
"""

import os
import sys
import math
import numpy as np
import rospy
import message_filters
from nav_msgs.msg import Odometry

# Non-interactive mode for matplotlib (avoids error when without display)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ------------------------------------------------------------------ #
# Data Collection
# ------------------------------------------------------------------ #

gt_traj   = []   # [(t, x, y, z, qx, qy, qz, qw), ...]
est_traj  = []   # Same as above

def sync_callback(gt_msg, est_msg):
    t = gt_msg.header.stamp.to_sec()

    gp = gt_msg.pose.pose.position
    gq = gt_msg.pose.pose.orientation
    gt_traj.append([t, gp.x, gp.y, gp.z, gq.x, gq.y, gq.z, gq.w])

    ep = est_msg.pose.pose.position
    eq = est_msg.pose.pose.orientation
    est_traj.append([t, ep.x, ep.y, ep.z, eq.x, eq.y, eq.z, eq.w])

    if len(gt_traj) % 50 == 0:
        rospy.loginfo(f'[eval] Recorded {len(gt_traj)} frames')


# ------------------------------------------------------------------ #
# Utility Functions
# ------------------------------------------------------------------ #

def quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))


def umeyama_align(src, dst):
    """
    Umeyama similitude transform alignment (translation + rotation only, no scale)
    src, dst: (N, 2) numpy array
    Returns aligned src
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1, d])
    R = Vt.T @ S @ U.T
    t = mu_d - R @ mu_s
    return (R @ src.T).T + t


def compute_ate(gt_xy, est_xy_aligned):
    """Point-wise Euclidean distance"""
    diff  = gt_xy - est_xy_aligned
    dists = np.linalg.norm(diff, axis=1)
    return dists


def compute_rpe(gt_xy, est_xy, interval=10):
    """
    Relative Pose Error (RPE)
    Compares the difference of relative translation every `interval` frames
    """
    errors = []
    n = len(gt_xy)
    for i in range(0, n - interval, interval):
        gt_delta  = gt_xy[i+interval]  - gt_xy[i]
        est_delta = est_xy[i+interval] - est_xy[i]
        errors.append(np.linalg.norm(gt_delta - est_delta))
    return np.array(errors)


def total_distance(xy):
    """Total trajectory distance"""
    if len(xy) < 2:
        return 0.0
    diffs = np.diff(xy, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


# ------------------------------------------------------------------ #
# Output Report and Images
# ------------------------------------------------------------------ #

def generate_report(output_dir):
    if len(gt_traj) < 10:
        rospy.logwarn('[eval] Insufficient data, at least 10 frames are required.')
        return

    gt  = np.array(gt_traj)   # (N, 8)
    est = np.array(est_traj)  # (N, 8)

    gt_xy  = gt[:, 1:3]    # x, y
    est_xy = est[:, 1:3]

    # ---- Alignment (Umeyama) ----
    est_xy_aligned = umeyama_align(est_xy, gt_xy)

    # ---- ATE ----
    ate_dists = compute_ate(gt_xy, est_xy_aligned)
    ate_mean  = float(np.mean(ate_dists))
    ate_rmse  = float(np.sqrt(np.mean(ate_dists**2)))
    ate_max   = float(np.max(ate_dists))
    ate_std   = float(np.std(ate_dists))

    # ---- RPE ----
    rpe_errors = compute_rpe(gt_xy, est_xy_aligned, interval=10)
    rpe_mean   = float(np.mean(rpe_errors)) if len(rpe_errors) > 0 else 0.0
    rpe_rmse   = float(np.sqrt(np.mean(rpe_errors**2))) if len(rpe_errors) > 0 else 0.0

    # ---- Distance & Drift Rate ----
    dist_total = total_distance(gt_xy)
    drift_rate = (ate_rmse / dist_total * 100) if dist_total > 0 else 0.0

    # ---- Time Stats ----
    duration = gt[-1, 0] - gt[0, 0]
    n_frames = len(gt)

    # ================================================================ #
    # Plotting
    # ================================================================ #
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('FAST-LIO2 vs CARLA Ground Truth', fontsize=15, fontweight='bold')

    # ---- Left Pane: Trajectory Comparison ----
    ax1 = axes[0]
    ax1.plot(gt_xy[:, 0],          gt_xy[:, 1],
             color='red',   linewidth=1.8, label='Ground Truth (CARLA)', zorder=3)
    ax1.plot(est_xy_aligned[:, 0], est_xy_aligned[:, 1],
             color='blue',  linewidth=1.5, label='FAST-LIO2 (aligned)',  zorder=2,
             linestyle='--')

    # Start and End Markers
    ax1.scatter(*gt_xy[0],    color='green',  s=100, zorder=5, marker='o', label='Start')
    ax1.scatter(*gt_xy[-1],   color='black',  s=100, zorder=5, marker='s', label='End')

    # Error lines (draw line every 20 frames or dynamically based on length)
    step = max(1, len(gt_xy) // 30)
    for i in range(0, len(gt_xy), step):
        ax1.plot([gt_xy[i, 0], est_xy_aligned[i, 0]],
                 [gt_xy[i, 1], est_xy_aligned[i, 1]],
                 color='orange', linewidth=0.6, alpha=0.5, zorder=1)

    orange_line = mpatches.Patch(color='orange', alpha=0.5, label='Position error')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles + [orange_line], fontsize=9)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory Comparison (Top View)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ---- Right Pane: ATE over Time ----
    ax2 = axes[1]
    times = gt[:, 0] - gt[0, 0]   # Relative time (s)
    ax2.plot(times, ate_dists, color='purple', linewidth=1.2, label='ATE per frame')
    ax2.axhline(ate_mean, color='red',    linewidth=1.5,
                linestyle='--', label=f'Mean ATE: {ate_mean:.3f}m')
    ax2.axhline(ate_rmse, color='orange', linewidth=1.5,
                linestyle=':',  label=f'RMSE ATE: {ate_rmse:.3f}m')
    ax2.fill_between(times, ate_dists, alpha=0.2, color='purple')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('ATE (m)')
    ax2.set_title('Absolute Trajectory Error over Time')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    img_path = os.path.join(output_dir, 'trajectory_compare.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    rospy.loginfo(f'[eval] Image saved: {img_path}')

    # ================================================================ #
    # Text Report
    # ================================================================ #
    report = f"""
==========================================================
         FAST-LIO2 Trajectory Evaluation Report
==========================================================

[Basic Information]
  Frames Recorded : {n_frames}
  Duration        : {duration:.1f} s
  Total Distance  : {dist_total:.2f} m

[ATE (Absolute Trajectory Error)] -- Frame-wise position error after alignment
  Mean  : {ate_mean:.4f} m
  RMSE  : {ate_rmse:.4f} m
  Max   : {ate_max:.4f} m
  Std   : {ate_std:.4f} m

[RPE (Relative Pose Error)] -- Relative shift error over 10 frames
  Mean  : {rpe_mean:.4f} m
  RMSE  : {rpe_rmse:.4f} m

[Drift Rate]
  ATE RMSE / Total Dist : {drift_rate:.3f} %
  (< 0.5% Excellent, < 1% Good, > 2% Poor)

[Conclusion]
"""

    if drift_rate < 0.5:
        report += '  ✓ Localization accuracy is excellent, suitable for parking planning.\n'
    elif drift_rate < 1.0:
        report += '  ✓ Localization accuracy is good, acceptable for parking planning.\n'
    elif drift_rate < 2.0:
        report += '  ^ Localization accuracy is fair, suggest checking IMU noise parameters.\n'
    else:
        report += '  x Large localization drift, need to inspect timestamp sync or extrinsics.\n'

    report += '=' * 58 + '\n'

    print(report)

    txt_path = os.path.join(output_dir, 'trajectory_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report)
    rospy.loginfo(f'[eval] Report saved: {txt_path}')


# ------------------------------------------------------------------ #
# Main Function
# ------------------------------------------------------------------ #

def main():
    rospy.init_node('trajectory_evaluator')

    output_dir = rospy.get_param('~output_dir',
                                 os.path.expanduser('~/trajectory_eval'))
    os.makedirs(output_dir, exist_ok=True)

    gt_topic  = rospy.get_param('~gt_topic',
                                '/carla/ego_vehicle/odometry')
    est_topic = rospy.get_param('~est_topic',
                                '/Odometry_fastlio')
    slop      = rospy.get_param('~sync_slop', 0.05)   # Time sync tolerance (s)

    rospy.loginfo(f'[eval] Ground Truth Topic : {gt_topic}')
    rospy.loginfo(f'[eval] Estimator Topic    : {est_topic}')
    rospy.loginfo(f'[eval] Output Directory   : {output_dir}')
    rospy.loginfo('[eval] Recording started. Press Ctrl+C to generate the report...')

    gt_sub  = message_filters.Subscriber(gt_topic,  Odometry)
    est_sub = message_filters.Subscriber(est_topic, Odometry)

    sync = message_filters.ApproximateTimeSynchronizer(
        [gt_sub, est_sub], queue_size=100, slop=slop)
    sync.registerCallback(sync_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rospy.loginfo(f'[eval] Recording finished. Total {len(gt_traj)} frames. Generating report...')
        generate_report(output_dir)


if __name__ == '__main__':
    main()
