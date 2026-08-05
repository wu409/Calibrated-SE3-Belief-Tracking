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
depth_dir = "./datasets/YCBInEOAT/bleach0/depth/"

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
depth_files = sorted(glob.glob(depth_dir + "/*.png"))


photo_features = []
actual_pose_errors = []

for i in range(min(len(pred_files), len(gt_files))):
    pred_pose = np.loadtxt(pred_files[i]) 
    gt_pose = np.loadtxt(gt_files[i])     
    
    adi_error = U.adi(pred_pose, gt_pose, open3d_model) * 100 
    actual_pose_errors.append(adi_error)

    depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
    depth_real = depth_raw.astype(np.float32) / 1000.0

    # PyRender 渲染彩色图
    pose_render = cv_to_gl @ pred_pose
    scene.set_pose(mesh_node, pose_render)
    color_render, depth_render = renderer.render(scene)
    
    pred_mask = (depth_render > 1e-6)

    # 🌟【关键修复 3】：相机照片 BGR 转 RGB，且去均值对齐！
    rgb_raw = cv2.imread(rgb_files[i], cv2.IMREAD_COLOR)
    rgb_real = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB) # BGR -> RGB
    
    gray_render = cv2.cvtColor(color_render, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_real = cv2.cvtColor(rgb_real, cv2.COLOR_RGB2GRAY).astype(np.float32)

    valid_mask = pred_mask | (depth_real > 1e-6)  

    if np.sum(valid_mask) > 0:
        r_pixels = gray_render[valid_mask]
        a_pixels = gray_real[valid_mask]
        # 去均值光度计算
        r_norm = r_pixels - np.mean(r_pixels)
        a_norm = a_pixels - np.mean(a_pixels)
        
        x4_photo = np.mean(np.abs(r_norm - a_norm))
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

plt.title('Photometric Residual vs. Actual Pose Error', fontsize=11)
plt.xlabel('Photometric Residual x_4', fontsize=11)
plt.ylabel('Actual ADD-S Pose Error (cm)', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('3-Photometric_Residual.png', dpi=300, bbox_inches='tight')
print("实打实的光度残差诊断图已成功生成: 3-Photometric_Residual.png！")







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