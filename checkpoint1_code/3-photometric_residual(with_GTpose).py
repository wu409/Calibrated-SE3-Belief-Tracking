import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pyrender
import Utils as U

# 相机内参 K 矩阵
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

res_dir = "./results/bleach0/"
data_dir = "./datasets/YCBInEOAT/bleach0/"
rgb_dir = "./datasets/YCBInEOAT/bleach0/rgb/"      
mesh_path  = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/textured.obj"
point_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz"

# 🌟【关键修复 1】：给 Scene 加上环境光 [1.0, 1.0, 1.0]！
scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0])

mesh = trimesh.load(mesh_path, process=False)
render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
mesh_node = scene.add(render_mesh)

camera = pyrender.IntrinsicsCamera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
camera_node = scene.add(camera, pose=np.eye(4))

# 🌟【关键修复 2】：加上方向光光源！
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=np.eye(4))

renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

with open(point_path, 'r') as f:
    model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))

pred_files = sorted(glob.glob(res_dir + "/*.txt"))
gt_files = sorted(glob.glob(data_dir + "/annotated_poses/*.txt"))
rgb_files = sorted(glob.glob(rgb_dir + "/*.png"))

mask_files = sorted(glob.glob(data_dir + "/gt_mask/*.png"))

photo_features = []
actual_pose_errors = []

for i in range(min(len(pred_files), len(gt_files))):
    pred_pose = np.loadtxt(pred_files[i]) 
    gt_pose = np.loadtxt(gt_files[i])     
    
    adi_error = U.adi(pred_pose, gt_pose, open3d_model) * 100 
    actual_pose_errors.append(adi_error)
    
   # 🌟 1. 渲染【真值姿态 T_gt】下的完美虚拟照片 (作为真值参照)
    pose_gt_gl = cv_to_gl @ gt_pose
    scene.set_pose(mesh_node, pose_gt_gl)
    color_gt, depth_gt = renderer.render(scene)

    # 🌟 2. 渲染【预测姿态 T_obs】下的预测虚拟照片
    pose_pred_gl = cv_to_gl @ pred_pose
    scene.set_pose(mesh_node, pose_pred_gl)
    color_pred, depth_pred = renderer.render(scene)

    # 🌟 3. 在同一个渲染域下，计算两者的纯光度残差 (彻底消除相机噪点和曝光干扰！)
    mask = (depth_gt > 1e-6) | (depth_pred > 1e-6)
    
    if np.sum(mask) > 0:
        # 转灰度图对比
        gray_gt = cv2.cvtColor(color_gt, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_pred = cv2.cvtColor(color_pred, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # 纯几何偏差引起的光度残差！
        x4_photo = np.mean(np.abs(gray_gt[mask] - gray_pred[mask]))
    else:
        x4_photo = 100.0

    photo_features.append(x4_photo)

renderer.delete()

# ==================== 3. 绘制光度残差诊断图 ====================
plt.figure(figsize=(8, 6))
plt.scatter(photo_features, actual_pose_errors, alpha=0.5, color='crimson', edgecolors='none', label=f'Frames (N={len(photo_features)})')

# 拟合正相关趋势线
z = np.polyfit(photo_features, actual_pose_errors, 1)
p = np.poly1d(z)
plt.plot(np.unique(photo_features), p(np.unique(photo_features)), "r--", linewidth=2, label='Fitted Trend (Positive Correlation)')

plt.title('Photometric Residual (using_GTpose) vs. Actual Pose Error', fontsize=11)
plt.xlabel('Photometric Residual x_4', fontsize=11)
plt.ylabel('Actual ADD-S Pose Error (cm)', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('3-Photometric_Residual(using_GTpose).png', dpi=300, bbox_inches='tight')
print("实打实的光度残差诊断图已成功生成: 3-Photometric_Residual(using_GTpose).png！")







# import os
# import glob
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import trimesh
# import pyrender
# import Utils as U


# # ==========================
# # 1. 路径
# # ==========================

# res_dir = "./results/bleach0/"
# data_dir = "./datasets/YCBInEOAT/bleach0/"

# rgb_dir = "./datasets/YCBInEOAT/bleach0/rgb/"

# mesh_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/textured.obj"

# point_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz"


# # ==========================
# # 2. 相机内参
# # ==========================

# K = np.array([
#     [3.195820007324218750e+02, 0.0, 3.202149847676955687e+02],
#     [0.0, 4.171186828613281250e+02, 2.443486680871046701e+02],
#     [0.0,0.0,1.0]
# ],dtype=np.float64)



# # ==========================
# # 3. pyrender初始化
# # ==========================

# print("loading mesh...")


# mesh = trimesh.load(
#     mesh_path,
#     process=False
# )


# render_mesh = pyrender.Mesh.from_trimesh(
#     mesh,
#     smooth=False
# )


# scene = pyrender.Scene()

# mesh_node = scene.add(render_mesh)



# camera = pyrender.IntrinsicsCamera(
#     fx=K[0,0],
#     fy=K[1,1],
#     cx=K[0,2],
#     cy=K[1,2]
# )


# camera_node = scene.add(
#     camera,
#     pose=np.eye(4)
# )



# renderer = pyrender.OffscreenRenderer(
#     viewport_width=640,
#     viewport_height=480
# )



# # ==========================
# # 4. 坐标转换
# # ==========================

# cv_to_gl = np.array([
#     [1,0,0,0],
#     [0,-1,0,0],
#     [0,0,-1,0],
#     [0,0,0,1]
# ])



# # ==========================
# # 5. ADD-S计算需要
# # ==========================

# with open(point_path,'r') as f:

#     model_pts=np.array(
#         [
#             list(map(float,line.split()))
#             for line in f.readlines()
#         ]
#     )


# open3d_model = U.toOpen3dCloud(
#     model_pts,
#     colors=np.zeros_like(model_pts)
# )



# # ==========================
# # 6. 文件
# # ==========================

# pred_files=sorted(
#     glob.glob(res_dir+"/*.txt")
# )


# gt_files=sorted(
#     glob.glob(data_dir+"/annotated_poses/*.txt")
# )



# rgb_files=sorted(
#     glob.glob(rgb_dir+"/*.png")
# )



# print(
#     "pred:",
#     len(pred_files),
#     "rgb:",
#     len(rgb_files)
# )



# # ==========================
# # 7. feature
# # ==========================


# photo_residuals=[]

# actual_pose_errors=[]



# # ==========================
# # 8. 逐帧计算
# # ==========================


# for i in range(
#     min(
#         len(pred_files),
#         len(gt_files),
#         len(rgb_files)
#     )
# ):


#     print("processing frame",i)



#     # ----------------------
#     # pose
#     # ----------------------

#     pred_pose=np.loadtxt(
#         pred_files[i]
#     )


#     gt_pose=np.loadtxt(
#         gt_files[i]
#     )


#     # ADD-S
#     error=U.adi(
#         pred_pose,
#         gt_pose,
#         open3d_model
#     )*100


#     actual_pose_errors.append(
#         error
#     )



#     # ----------------------
#     # 读取真实RGB
#     # ----------------------

#     rgb=cv2.imread(
#         rgb_files[i]
#     )


#     if rgb is None:
#         print(
#             "rgb missing:",
#             rgb_files[i]
#         )
#         continue



#     rgb=cv2.cvtColor(
#         rgb,
#         cv2.COLOR_BGR2RGB
#     )


#     rgb_real=rgb.astype(
#         np.float32
#     )/255.0



#     # ----------------------
#     # 设置物体pose
#     # ----------------------


#     pose_render = cv_to_gl @ pred_pose


#     scene.set_pose(
#         mesh_node,
#         pose_render
#     )



#     # ----------------------
#     # 渲染RGB + depth
#     # ----------------------

#     rgb_render, depth_render = renderer.render(
#         scene
#     )



#     rgb_render=rgb_render.astype(
#         np.float32
#     )/255.0



#     # ----------------------
#     # render mask
#     # ----------------------

#     render_mask = depth_render > 0



#     # ----------------------
#     # photometric residual
#     # ----------------------


#     valid = render_mask


#     if valid.sum()>20:


#         diff=np.abs(
#             rgb_render[valid]
#             -
#             rgb_real[valid]
#         )


#         # RGB三个通道平均
#         photo_error=np.mean(
#             diff
#         )


#     else:

#         photo_error=1.0


#     if i == 0:   # 只保存第一帧，避免循环保存663张
    
#         cv2.imwrite(
#             "2-rgb_real.png",
#             (rgb_render[valid] * 255).astype(np.uint8)
#         )

#         cv2.imwrite(
#             "2-rgb_real.png",
#             (rgb_real[valid] * 255).astype(np.uint8)
#         )

#         print("mask saved!")

#     photo_residuals.append(
#         photo_error
#     )




# # ==========================
# # 9. 绘图
# # ==========================


# photo_residuals=np.array(
#     photo_residuals
# )


# actual_pose_errors=np.array(
#     actual_pose_errors[:len(photo_residuals)]
# )



# plt.figure(figsize=(8,6))


# plt.scatter(
#     photo_residuals,
#     actual_pose_errors,
#     alpha=0.5
# )



# # 趋势线

# if len(photo_residuals)>2:

#     z=np.polyfit(
#         photo_residuals,
#         actual_pose_errors,
#         1
#     )

#     p=np.poly1d(z)


#     xs=np.linspace(
#         photo_residuals.min(),
#         photo_residuals.max(),
#         100
#     )


#     plt.plot(
#         xs,
#         p(xs),
#         'r--',
#         linewidth=2
#     )



# plt.xlabel(
#     "Photometric Residual"
# )


# plt.ylabel(
#     "ADD-S Pose Error (cm)"
# )


# plt.title(
#     "Photometric Residual vs Pose Error"
# )


# plt.grid(
#     linestyle="--",
#     alpha=0.5
# )



# plt.savefig(
#     "3-photometric_residual_vs_pose_error.png",
#     dpi=300,
#     bbox_inches="tight"
# )



# print(
#     "done!"
# )

# print(
#     "mean photometric residual:",
#     photo_residuals.mean()
# )

# print(
#     "max:",
#     photo_residuals.max()
# )