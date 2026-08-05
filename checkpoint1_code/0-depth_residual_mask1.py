import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import Utils as U  
import trimesh
import pyrender

K = np.array([
    [3.195820007324218750e+02,    0.0,   3.202149847676955687e+02],
    [   0.0,  4.171186828613281250e+02, 2.443486680871046701e+02],
    [   0.0,      0.0,     1.0   ]
], dtype=np.float64)

cv_to_gl = np.array([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,1]
])

# ==================== 1. 本地路径配置 ====================
res_dir = "./results_collection/bleach0/bleach0_occ40/"                             # 预测出的 .txt 姿态路径
data_dir = "./datasets/YCBInEOAT/bleach0/"                # 数据集路径
depth_dir = "./datasets/YCBInEOAT_Corrupted/bleach0_occ40/depth/"         # 真实深度图路径
point_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz" # 3D模型路径
mesh_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/textured.obj"
mesh = trimesh.load(mesh_path)
render_mesh = pyrender.Mesh.from_trimesh(mesh,smooth=False)
scene = pyrender.Scene()
mesh_node = scene.add(render_mesh)

camera = pyrender.IntrinsicsCamera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
camera_node = scene.add(camera, pose=np.eye(4))
renderer = pyrender.OffscreenRenderer(viewport_width=640,viewport_height=480)



# ==================== 2. 载入 3D 模型点云 ====================
print("正在载入 3D CAD 点云模型...")
with open(point_path, 'r') as f:
    lines = f.readlines()
model_pts = np.array([list(map(float, line.rstrip().split())) for line in lines]) # (N, 3)
open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))

# 读取预测文件与真值文件
pred_files = sorted(glob.glob(res_dir + "/*.txt"))
gt_files = sorted(glob.glob(data_dir + "/annotated_poses/*.txt"))
depth_files = sorted(glob.glob(depth_dir + "/*.png")) # 或 .npy

real_depth_residuals = []
actual_pose_errors = []



# ==================== 2. 逐帧计算【真实深度残差 x_1】 ====================
for i in range(min(len(pred_files), len(gt_files))):
    pred_pose = np.loadtxt(pred_files[i]) 
    gt_pose = np.loadtxt(gt_files[i])     

    # ---------------- A. 姿态误差 (Y轴) ----------------
    adi_error = U.adi(pred_pose, gt_pose, open3d_model) * 100 
    actual_pose_errors.append(adi_error)


    # ---------------- B. 真实深度残差 (X轴) ----------------
    depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
    depth_real = depth_raw.astype(np.float32) / 1000.0

    pose_render = cv_to_gl @ pred_pose
    scene.set_pose(mesh_node,pose_render)
    depth_render = renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)

    
    valid = (depth_render>0)

    if np.sum(valid)>20:
        residual=np.abs(depth_render[valid]-depth_real[valid])
        # 平均深度误差
        x1_real=np.mean(residual)*100
        print(x1_real)
    else:
        # 完全没有匹配区域
        x1_real=20.0

    real_depth_residuals.append(x1_real)

# ==================== 4. 绘制并保存真实的诊断散点图 ====================
plt.figure(figsize=(8, 6))

plt.scatter(real_depth_residuals, actual_pose_errors, alpha=0.5, color='dodgerblue', edgecolors='none', label=f'Frames (N={len(real_depth_residuals)})')

# 拟合一条真实数据趋势线
z = np.polyfit(real_depth_residuals, actual_pose_errors, 1)
p = np.poly1d(z)
plt.plot(np.unique(real_depth_residuals), p(np.unique(real_depth_residuals)), "r--", linewidth=2, label='Fitted Correlation Trend')

plt.title('Depth Residual vs. Actual Pose Error', fontsize=12)
plt.xlabel('Depth Residual x_1 (cm)', fontsize=11)
plt.ylabel('Actual ADD-S Pose Error (cm)', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('0-Depth_Residual_vs_Pose_Error.png', dpi=300, bbox_inches='tight')
print("已成功从图像像素中算出真实残差！照片已保存为: 0-Depth_Residual_vs_Pose_Error.png")




