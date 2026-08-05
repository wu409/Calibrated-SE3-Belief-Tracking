import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import Utils as U  # 调用项目自带的工具函数

# ==================== 1. 本地路径配置 ====================
res_dir = "./results_collection/bleach0/bleach0_occ40/"      
data_dir = "./datasets/YCBInEOAT/bleach0/"
depth_dir = "./datasets/YCBInEOAT_Corrupted/bleach0_occ40/depth/"  
point_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz"

# YCBInEOAT 相机内参 K 矩阵
K = np.array([
    [3.195820007324218750e+02,    0.0,   3.202149847676955687e+02],
    [   0.0,  4.171186828613281250e+02, 2.443486680871046701e+02],
    [   0.0,      0.0,     1.0   ]
], dtype=np.float64)

# 2. 载入模型与路径
with open(point_path, 'r') as f:
    model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))

pred_files = sorted(glob.glob(res_dir + "/*.txt"))
gt_files = sorted(glob.glob(data_dir + "/annotated_poses/*.txt"))
depth_files = sorted(glob.glob(depth_dir + "/*.png"))

# mask_files = sorted(glob.glob(data_dir + "/masks/*.png"))
# if len(mask_files) == 0:
#     mask_files = sorted(glob.glob(data_dir + "/mask/*.png"))

print(f"开始构造内点率特征 (Inlier Score x_3)，处理 {len(pred_files)} 帧数据...")

inlier_scores = []
actual_pose_errors = []

# ==================== 3. 逐帧构造 x_3 并计算真实姿态误差 ====================
for i in range(min(len(pred_files), len(gt_files))):
    pred_pose = np.loadtxt(pred_files[i]) 
    gt_pose = np.loadtxt(gt_files[i])     
    
    # A. 真实 ADD-S 姿态误差 (Y轴, 单位: cm)
    adi_error = U.adi(pred_pose, gt_pose, open3d_model) * 100 
    actual_pose_errors.append(adi_error)
    
    # B. 构造特征 x_3: 几何深度内点率 (X轴)
    depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
    depth_real = depth_raw.astype(np.float32) / 1000.0 # 转换为米
    
    # 3D 点云投影
    R_pred, t_pred = pred_pose[:3, :3], pred_pose[:3, 3]
    pts_cam = (R_pred @ model_pts.T).T + t_pred 
    
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    u = np.round((K[0, 0] * X / Z) + K[0, 2]).astype(int)
    v = np.round((K[1, 1] * Y / Z) + K[1, 2]).astype(int)
    
    valid_bounds = (u >= 0) & (u < 640) & (v >= 0) & (v < 480) & (Z > 0)
    u_valid, v_valid, Z_pred = u[valid_bounds], v[valid_bounds], Z[valid_bounds]
    projected_mask = np.zeros((480,640), dtype=np.uint8)
    projected_mask[v_valid, u_valid] = 1
    Z_real = depth_real[v_valid, u_valid]
    valid_depth = Z_real > 0

    if valid_depth.sum()>0:
        Z_pred_valid = Z_pred[valid_depth]
        Z_real_valid = Z_real[valid_depth]

        depth_diff = np.abs(Z_pred_valid-Z_real_valid)*100
        x3_score = 1 - np.mean(depth_diff < 2.0)     
    else:
        x3_score = 1

    inlier_scores.append(x3_score)

# ==================== 4. 绘制独立诊断图 ====================
plt.figure(figsize=(8, 6))
plt.scatter(inlier_scores, actual_pose_errors, alpha=0.6, color='forestgreen', edgecolors='none', label=f'Frames (N={len(inlier_scores)})')

# 拟合趋势线
z = np.polyfit(inlier_scores, actual_pose_errors, 1)
p = np.poly1d(z)
plt.plot(np.unique(inlier_scores), p(np.unique(inlier_scores)), "r--", linewidth=2, label='Fitted Trend (Negative Correlation)')

plt.title(' Inlier Score vs. Actual Pose Error', fontsize=11)
plt.xlabel('Inlier Score (1-Inlier_Ratio) x_3 (2cm Depth Tolerance)', fontsize=11)
plt.ylabel('Actual ADD-S Pose Error (cm)', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('1-Inlier_Score_vs_Pose_Error.png', dpi=300, bbox_inches='tight')
print("内点率特征诊断图已成功生成: 1-Inlier_Score_vs_Pose_Error.png！")