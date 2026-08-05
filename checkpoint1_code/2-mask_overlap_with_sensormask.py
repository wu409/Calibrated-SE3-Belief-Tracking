import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pyrender
import Utils as U  # 调用项目自带工具函数



# 相机内参 K 矩阵 (用于 3D 投影)
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
res_dir = "./results/bleach0/"
data_dir = "./datasets/YCBInEOAT/bleach0/"
depth_dir = "./datasets/YCBInEOAT/bleach0/depth/"
mesh_path  = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/textured.obj"
point_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz"
mesh = trimesh.load(mesh_path)
render_mesh = pyrender.Mesh.from_trimesh(mesh,smooth=False)
scene = pyrender.Scene()
mesh_node = scene.add(render_mesh)

camera = pyrender.IntrinsicsCamera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
camera_node = scene.add(camera, pose=np.eye(4))
renderer = pyrender.OffscreenRenderer(viewport_width=640,viewport_height=480)


# 2. 载入模型与路径
with open(point_path, 'r') as f:
    model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))

pred_files = sorted(glob.glob(res_dir + "/*.txt"))
gt_files = sorted(glob.glob(data_dir + "/annotated_poses/*.txt"))
depth_files = sorted(glob.glob(depth_dir + "/*.png"))

print(f"开始运行【纯在线、零 GT 依赖】的 Mask IoU 诊断计算，处理 {len(pred_files)} 帧...")

mask_features = []
actual_pose_errors = []

# ==================== 3. 逐帧纯在线 Mask 计算 ====================
for i in range(min(len(pred_files), len(gt_files))):
    pred_pose = np.loadtxt(pred_files[i]) 
    gt_pose = np.loadtxt(gt_files[i])     
    
    # A. 真实姿态误差 (Y轴, cm)
    adi_error = U.adi(pred_pose, gt_pose, open3d_model) * 100 
    actual_pose_errors.append(adi_error)
    
    # B. 载入相机真实拍到的深度图 (单位: 米)
    depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
    depth_real = depth_raw.astype(np.float32) / 1000.0
    
    # 🌟【步骤 1】：生成预测的 2D 渲染掩码 pred_mask

    pose_render = cv_to_gl @ pred_pose
    scene.set_pose(mesh_node,pose_render)
    depth_render = renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)
    print(
    "frame",
    i,
    "depth max:",
    depth_render.max(),
    "valid pixels:",
    np.sum(depth_render>0)
    )
    pred_mask = depth_render > 1e-6

    # 🌟【步骤 2】：在相机深度图上提取纯在线传感器掩码 sensor_mask (零 GT 依赖！)
    z_center = pred_pose[2, 3]
    #sensor_mask = (depth_real > (z_center - 0.2)) & (depth_real < (z_center + 0.2))
    sensor_mask = depth_real>0

    # 🌟【步骤 3】：计算 1 - Online IoU (不重合残差特征 x_mask)
    intersection=np.logical_and(pred_mask,sensor_mask).sum()
    union = np.logical_or(pred_mask, sensor_mask).sum()

    iou = intersection / (union + 1e-6)
    x_mask = 1.0 - iou # 1 - IoU 越接近 0 越准，越接近 1 越差

    mask_features.append(x_mask)
    if i == 0:   # 只保存第一帧，避免循环保存663张

        cv2.imwrite(
            "2-render_mask(with_sensor_mask).png",
            (pred_mask * 255).astype(np.uint8)
        )

        cv2.imwrite(
            "2-sensor_mask(with_sensor_mask).png",
            (sensor_mask * 255).astype(np.uint8)
        )

        print("mask saved!")

# ==================== 4. 绘制诊断图 ====================
plt.figure(figsize=(8, 6))
plt.scatter(mask_features, actual_pose_errors, alpha=0.5, color='darkcyan', edgecolors='none', label=f'Frames (N={len(mask_features)})')

# 拟合正相关趋势线
z = np.polyfit(mask_features, actual_pose_errors, 1)
p = np.poly1d(z)
plt.plot(np.unique(mask_features), p(np.unique(mask_features)), "r--", linewidth=2, label='Fitted Trend (Positive Correlation)')

plt.title('Mask Overlap(with sensor_mask) vs. Actual Pose Error', fontsize=11)
plt.xlabel('Mask Overlap x_2 (1 - IoU)', fontsize=11)
plt.ylabel('Actual ADD-S Pose Error (cm)', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('2-Mask_Overlap(with sensor_mask).png', dpi=300, bbox_inches='tight')
print("纯在线零 GT 的 Mask 诊断图已成功生成: 2-Mask_Overlap(with sensor_mask).png！")