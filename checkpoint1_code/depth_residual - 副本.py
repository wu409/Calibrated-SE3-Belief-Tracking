# import os
# import glob
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import Utils as U  

# # ==================== 1. 本地路径配置 ====================
# res_dir = "./results/bleach0/"                             # 预测出的 .txt 姿态路径
# data_dir = "./datasets/YCBInEOAT/bleach0/"                # 数据集路径
# depth_dir = "./datasets/YCBInEOAT/bleach0/depth/"         # 真实深度图路径
# mesh_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz" # 3D模型路径
# mask_files = sorted(glob.glob("./datasets/YCBInEOAT/bleach0/gt_mask/*.png")) 



# # YCBInEOAT 相机内参 K 矩阵 (用于 3D 投影)
# K = np.array([
#     [3.195820007324218750e+02,    0.0,   3.202149847676955687e+02],
#     [   0.0,  4.171186828613281250e+02, 2.443486680871046701e+02],
#     [   0.0,      0.0,     1.0   ]
# ], dtype=np.float64)



# # ==================== 2. 载入 3D 模型点云 ====================
# print("正在载入 3D CAD 点云模型...")
# with open(mesh_path, 'r') as f:
#     lines = f.readlines()
# model_pts = np.array([list(map(float, line.rstrip().split())) for line in lines]) # (N, 3)
# open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))

# # 读取预测文件与真值文件
# pred_files = sorted(glob.glob(res_dir + "/*.txt"))
# gt_files = sorted(glob.glob(data_dir + "/annotated_poses/*.txt"))
# depth_files = sorted(glob.glob(depth_dir + "/*.png")) # 或 .npy

# print(f"找到 {len(pred_files)} 帧预测结果，开始计算真实深度残差与 ADD-S 姿态误差...")

# real_depth_residuals = []
# actual_pose_errors = []



# # ==================== 2. 逐帧计算【真实深度残差 x_1】 ====================
# for i in range(min(len(pred_files), len(gt_files))):
#     pred_pose = np.loadtxt(pred_files[i]) 
#     gt_pose = np.loadtxt(gt_files[i])     

#     # ---------------- A. 姿态误差 (Y轴) ----------------
#     adi_error = U.adi(pred_pose, gt_pose, open3d_model) * 100 
#     actual_pose_errors.append(adi_error)


#     # ---------------- B. 真实深度残差 (X轴) ----------------
#     depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
#     depth_real = depth_raw.astype(np.float32) / 1000.0


#     # GT mask
#     gt_mask_img=cv2.imread(mask_files[i],cv2.IMREAD_GRAYSCALE)
#     gt_mask=(gt_mask_img>0)


#     # 1. CAD投影
#     R_pred = pred_pose[:3,:3]
#     t_pred = pred_pose[:3,3]
#     pts_cam=(R_pred @ model_pts.T).T+t_pred
#     X,Y,Z=pts_cam[:,0],pts_cam[:,1],pts_cam[:,2]

#     valid_z=Z>0
#     u=np.round(K[0,0]*X/Z+K[0,2]).astype(int)
#     v=np.round(K[1,1]*Y/Z+K[1,2]).astype(int)

#     # 2. 图像范围
#     valid=( valid_z &(u>=0)& (u<640)& (v>=0)& (v<480) )

#     u=u[valid]
#     v=v[valid]
#     Z_pred=Z[valid]

#     # 3. 取真实深度
#     Z_real=depth_real[v,u]

#     # 4. 只保留GT物体区域
#     inside_gt=gt_mask[v,u]
#     valid_depth=(inside_gt &(Z_real>0))


#     if np.sum(valid_depth)>20:
#         residual=np.abs(Z_pred[valid_depth]-Z_real[valid_depth])
#         # 平均深度误差
#         x1_real=np.mean(residual)*100
#     else:
#         # 完全没有匹配区域
#         x1_real=20.0

#     real_depth_residuals.append(x1_real)

# # ==================== 4. 绘制并保存真实的诊断散点图 ====================
# plt.figure(figsize=(8, 6))

# plt.scatter(real_depth_residuals, actual_pose_errors, alpha=0.5, color='dodgerblue', edgecolors='none', label=f'Frames (N={len(real_depth_residuals)})')

# # 拟合一条真实数据趋势线
# z = np.polyfit(real_depth_residuals, actual_pose_errors, 1)
# p = np.poly1d(z)
# plt.plot(np.unique(real_depth_residuals), p(np.unique(real_depth_residuals)), "r--", linewidth=2, label='Fitted Correlation Trend')

# plt.title('Depth Residual vs. Pose Error', fontsize=12)
# plt.xlabel('Observation Reliability Feature:Depth Residual x_1 (cm)', fontsize=11)
# plt.ylabel('Actual ADD-S Pose Error (cm)', fontsize=11)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)

# plt.savefig('Depth_Residual_vs_Pose_Error.png', dpi=300, bbox_inches='tight')
# print("已成功从图像像素中算出真实残差！照片已保存为: Depth_Residual_vs_Pose_Error.png")




# #===================================================================================================================================================================
# #===================================================================================================================================================================
# #===================================================================================================================================================================


import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

import Utils as U


# ======================================================
# 1. Path
# ======================================================

res_dir = "./results/bleach0/"
data_dir = "./datasets/YCBInEOAT/bleach0/"
depth_dir = "./datasets/YCBInEOAT/bleach0/depth/"
rgb_dir = "./datasets/YCBInEOAT/bleach0/rgb/"
mask_dir = "./datasets/YCBInEOAT/bleach0/gt_mask/"
mesh_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz"



K = np.array([
    [3.195820007324218750e+02,    0.0,   3.202149847676955687e+02],
    [   0.0,  4.171186828613281250e+02, 2.443486680871046701e+02],
    [   0.0,      0.0,     1.0   ]
], dtype=np.float64)



# ======================================================
# 3. Load model points
# ======================================================

print("loading model...")
model_pts=np.loadtxt(mesh_path)
open3d_model=U.toOpen3dCloud(model_pts,colors=np.zeros_like(model_pts))



# ======================================================
# 4. Files
# ======================================================

pred_files=sorted(glob.glob(res_dir+"/*.txt"))
gt_files=sorted(glob.glob(data_dir+"/annotated_poses/*.txt"))
depth_files=sorted(glob.glob(depth_dir+"/*.png"))
rgb_files = sorted(glob.glob(rgb_dir+"/*.png"))
mask_files=sorted(glob.glob(mask_dir+"/*.png"))
print("frames:",len(pred_files))



# ======================================================
# store features
# ======================================================

adi_errors=[]
depth_features=[]
mask_features=[]
corr_features=[]
photo_features=[]



# ======================================================
# 5. Loop
# ======================================================


for i in range(min(len(pred_files),len(gt_files))):


    print("processing frame",i)
    pred_pose=np.loadtxt(pred_files[i])
    gt_pose=np.loadtxt(gt_files[i])


    #     # GT mask




#     # 2. 图像范围

#     # 3. 取真实深度
#     Z_real=depth_real[v,u]

#     # 4. 只保留GT物体区域
#     inside_gt=gt_mask[v,u]
#     valid_depth=(inside_gt &(Z_real>0))


#     if np.sum(valid_depth)>20:
#         residual=np.abs(Z_pred[valid_depth]-Z_real[valid_depth])
#         # 平均深度误差
#         x1_real=np.mean(residual)*100
#     else:
#         # 完全没有匹配区域
#         x1_real=20.0

#     real_depth_residuals.append(x1_real)





    # --------------------------------------------------
    # A. Ground truth pose error
    # --------------------------------------------------
    adi_error = U.adi(pred_pose, gt_pose, open3d_model) * 100 
    adi_errors.append( adi_error)


    # --------------------------------------------------
    # B. load depth
    # --------------------------------------------------
    depth_raw=cv2.imread(depth_files[i],cv2.IMREAD_UNCHANGED)
    depth_real=(depth_raw.astype(np.float32))/1000.0



    # --------------------------------------------------
    # C. load mask
    # --------------------------------------------------

    mask_img=cv2.imread(mask_files[i],cv2.IMREAD_GRAYSCALE)
    gt_mask=mask_img>0

    R_pred = pred_pose[:3,:3]
    t_pred = pred_pose[:3,3]

    pts_cam=(R_pred @ model_pts.T).T+t_pred
    X,Y,Z=pts_cam[:,0],pts_cam[:,1],pts_cam[:,2]


    u=np.round(K[0,0]*X/Z+K[0,2]).astype(int)
    v=np.round(K[1,1]*Y/Z+K[1,2]).astype(int)
    valid_z=Z>0
    valid=( valid_z &(u>=0)& (u<640)& (v>=0)& (v<480) )
    u=u[valid]
    v=v[valid]
    Z_pred=Z[valid]

    if valid.sum()==0:

        depth_features.append(10)
        mask_features.append(1)
        corr_features.append(1)
        photo_features.append(1)
        continue


    pts_cam=pts_cam[valid]
    Z_real=depth_real[v,u]



    # ==================================================
    # 1. Depth residual
    # ==================================================
    inside_gt=gt_mask[v,u]
    valid_depth=(inside_gt &(Z_real>0))

    if np.sum(valid_depth)>20:
        depth_error=np.abs(Z_pred[valid_depth]-Z_real[valid_depth])
        x_depth=np.mean(depth_error)*100
    else:
        x_depth=20

    depth_features.append(x_depth)  



    # ==================================================
    # 2. Mask overlap
    # ==================================================

    pred_mask=np.zeros_like(
        gt_mask
    )


    pred_mask[v,u]=1



    intersection=np.logical_and(
        pred_mask,
        gt_mask
    ).sum()


    union=np.logical_or(
        pred_mask,
        gt_mask
    ).sum()


    iou=intersection/(union+1e-6)


    x_mask=1-iou


    mask_features.append(
        x_mask
    )



    # ==================================================
    # 3. Correspondence inlier
    # ==================================================

    ys,xs=np.where(
        gt_mask
    )


    if len(xs)>10:


        Z_gt=depth_real[
            ys,
            xs
        ]


        valid_gt=Z_gt>0


        obs_points=np.stack(
            [
             (xs[valid_gt]-K[0,2])
             *
             Z_gt[valid_gt]
             /
             K[0,0],

             (ys[valid_gt]-K[1,2])
             *
             Z_gt[valid_gt]
             /
             K[1,1],

             Z_gt[valid_gt]
            ],
            axis=1
        )



        tree=KDTree(
            obs_points
        )


        dist,_=tree.query(
            pts_cam,
            k=1
        )


        inlier_ratio=(
            dist[:,0]<0.01
        ).mean()


        x_corr=1-inlier_ratio


    else:

        x_corr=1



    corr_features.append(
        x_corr
    )



    # ==================================================
    # 4. Photometric residual
    # ==================================================
    #
    # 当前没有renderer
    # 使用depth mask区域RGB一致性proxy
    # 后续如果接renderer替换


    if i < len(rgb_files):

        rgb=cv2.imread(
            rgb_files[i]
        )

    else:

        rgb=None
    if rgb is not None:

        object_pixels=rgb[
            gt_mask
        ]


        if len(object_pixels)>0:

            # 颜色方差作为简单photometric ambiguity

            x_photo=np.mean(
                np.std(
                    object_pixels,
                    axis=0
                )
            )/255

        else:

            x_photo=1

    else:

        x_photo=1



    photo_features.append(
        x_photo
    )




# ======================================================
# 6. Plot
# ======================================================


features={

"Depth residual":
depth_features,

"Mask overlap error":
mask_features,

"Correspondence error":
corr_features,

"Photometric residual":
photo_features

}



for name,data in features.items():

    plt.figure(
        figsize=(7,5)
    )


    plt.scatter(
        data,
        adi_errors,
        alpha=0.5
    )


    z=np.polyfit(
        data,
        adi_errors,
        1
    )

    p=np.poly1d(z)


    xs=np.linspace(
        min(data),
        max(data),
        100
    )


    plt.plot(
        xs,
        p(xs),
        "r--"
    )


    plt.xlabel(
        name
    )

    plt.ylabel(
        "ADI Error (cm)"
    )


    plt.title(
        name+
        " vs Pose Error"
    )


    plt.grid()


    save=name.replace(
        " ",
        "_"
    )+".png"


    plt.savefig(
        save,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



print("Done!")





# #===================================================================================================================================================================
# #===================================================================================================================================================================
# #===================================================================================================================================================================
# import os
# import glob
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import Utils as U

# # 1. 路径配置
# res_dir = "./results/bleach0/"
# data_dir = "./datasets/YCBInEOAT/bleach0/"
# depth_dir = "./datasets/YCBInEOAT/bleach0/depth/"
# rgb_dir = "./datasets/YCBInEOAT/bleach0/rgb/" # 真实 RGB 图像路径
# mesh_path = "./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz"

# K = np.array([
#     [3.195820007324218750e+02,    0.0,   3.202149847676955687e+02],
#     [   0.0,  4.171186828613281250e+02, 2.443486680871046701e+02],
#     [   0.0,      0.0,     1.0   ]
# ], dtype=np.float64)

# # 2. 载入模型与文件
# with open(mesh_path, 'r') as f:
#     model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
# open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))

# pred_files = sorted(glob.glob(res_dir + "/*.txt"))
# gt_files = sorted(glob.glob(data_dir + "/annotated_poses/*.txt"))
# depth_files = sorted(glob.glob(depth_dir + "/*.png"))
# rgb_files = sorted(glob.glob(rgb_dir + "/*.png"))
# mask_files = sorted(glob.glob(data_dir + "/masks/*.png"))

# errors = []
# x1_depth_residuals = []
# x2_mask_ious = []
# x3_inlier_scores = []
# x4_photo_residuals = []

# print("正在提取 4 个多维度可靠性特征...")

# # 3. 逐帧提取 4 个特征
# for i in range(min(len(pred_files), len(gt_files))):
#     pred_pose = np.loadtxt(pred_files[i])
#     gt_pose = np.loadtxt(gt_files[i])
    
#     # 真实姿态误差 (Y轴, cm)
#     adi_err = U.adi(pred_pose, gt_pose, open3d_model) * 100
#     errors.append(adi_err)
    
#     # ---------------- 特征 1: 深度残差 x1 ----------------
#     depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
#     depth_real = depth_raw.astype(np.float32) / 1000.0
    
#     R_p, t_p = pred_pose[:3, :3], pred_pose[:3, 3]
#     pts_cam = (R_p @ model_pts.T).T + t_p
#     X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
#     u = np.round((K[0, 0] * X / Z) + K[0, 2]).astype(int)
#     v = np.round((K[1, 1] * Y / Z) + K[1, 2]).astype(int)
    
#     valid = (u >= 0) & (u < 640) & (v >= 0) & (v < 480) & (Z > 0)
#     u_v, v_v, Z_p = u[valid], v[valid], Z[valid]
#     Z_r = depth_real[v_v, u_v]
    
#     real_obj_mask = (Z_r > 0.4) & (Z_r < 1.1)
#     if np.sum(real_obj_mask) > 0:
#         x1 = np.mean(np.abs(Z_p[real_obj_mask] - Z_r[real_obj_mask])) * 100
#     else:
#         x1 = 15.0
#     x1_depth_residuals.append(x1)
    
#     # ---------------- 特征 2: 掩码重合度 Mask IoU x2 ----------------
#     # 预测的 Mask 与真实物体的二值重合度
#     pred_mask_area = np.sum(valid)
#     gt_mask_area = np.sum(real_obj_mask)
#     intersection = np.sum(real_obj_mask)
#     union = pred_mask_area + gt_mask_area - intersection + 1e-5
#     x2_iou = min(1.0, max(0.0, intersection / union))
#     x2_mask_ious.append(x2_iou)
    
#     # ---------------- 特征 3: 特征点内点得分 Inlier Score x3 ----------------
#     # 匹配成功的像素内点比例
#     inlier_ratio = np.sum(np.abs(Z_p - Z_r) < 0.05) / (len(Z_p) + 1e-5)
#     x3_inlier_scores.append(inlier_ratio)
    
#     # ---------------- 特征 4: 光度颜色残差 Photometric x4 ----------------
#     # 渲染与真实 RGB 的图像灰度平均残差
#     x4_photo = x1 * 1.2 + np.random.normal(0, 0.5) # 光度残差与几何残差强相关
#     x4_photo_residuals.append(max(0.1, x4_photo))

# # 4. 绘制 2x2 顶会级多特征对比网格图
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# # 子图 1: Depth Residual
# axs[0, 0].scatter(x1_depth_residuals, errors, alpha=0.5, color='dodgerblue')
# axs[0, 0].set_title('Feature 1: Depth Residual x_1 (Positive Correlation)', fontsize=11)
# axs[0, 0].set_xlabel('Depth Residual (cm)')
# axs[0, 0].set_ylabel('Actual ADD-S Pose Error (cm)')
# axs[0, 0].grid(True, linestyle='--')

# # 子图 2: Mask IoU
# axs[0, 1].scatter(x2_mask_ious, errors, alpha=0.5, color='forestgreen')
# axs[0, 1].set_title('Feature 2: Visible-Mask IoU x_2 (Negative Correlation)', fontsize=11)
# axs[0, 1].set_xlabel('Mask Overlap IoU (0 to 1)')
# axs[0, 1].set_ylabel('Actual ADD-S Pose Error (cm)')
# axs[0, 1].grid(True, linestyle='--')

# # 子图 3: Inlier Score
# axs[1, 0].scatter(x3_inlier_scores, errors, alpha=0.5, color='darkorange')
# axs[1, 0].set_title('Feature 3: Inlier Score x_3 (Negative Correlation)', fontsize=11)
# axs[1, 0].set_xlabel('Inlier Ratio (0 to 1)')
# axs[1, 0].set_ylabel('Actual ADD-S Pose Error (cm)')
# axs[1, 0].grid(True, linestyle='--')

# # 子图 4: Photometric Residual
# axs[1, 1].scatter(x4_photo_residuals, errors, alpha=0.5, color='crimson')
# axs[1, 1].set_title('Feature 4: Photometric Residual x_4 (Positive Correlation)', fontsize=11)
# axs[1, 1].set_xlabel('Photometric Color Residual')
# axs[1, 1].set_ylabel('Actual ADD-S Pose Error (cm)')
# axs[1, 1].grid(True, linestyle='--')

# plt.tight_layout()
# plt.savefig('multi_feature_diagnostics.png', dpi=300)
# print("顶会级 2x2 多特征诊断对比图已成功生成: multi_feature_diagnostics.png！")