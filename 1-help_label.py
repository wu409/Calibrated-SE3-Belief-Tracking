import os
import glob
import numpy as np
import pandas as pd
import cv2
import Utils as U
from scipy.spatial.transform import Rotation as R_sci
import trimesh
import pyrender
import argparse

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

open3d_models = []

def se3_log_map(T):
    R_mat = T[:3, :3]
    t_vec = T[:3, 3]
    w_vec = R_sci.from_matrix(R_mat).as_rotvec() # 3D 旋转向量
    return np.concatenate([t_vec, w_vec])

# 2. SE(3) 指数映射: 把 6维向量 [平移(3), 旋转向量(3)] 转成 4x4 变换矩阵
def se3_exp_map(delta):
    t_vec = delta[:3]
    w_vec = delta[3:]
    R_mat = R_sci.from_rotvec(w_vec).as_matrix()
    
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mat
    T[:3, 3] = t_vec
    return T



def reliability_depth_residual(depth_real, pred_pose, scene, renderer,mesh_node):
    pose_render = cv_to_gl @ pred_pose
    scene.set_pose(mesh_node,pose_render)
    depth_render = renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)
    valid = (depth_render>0)
    if np.sum(valid)>20:
        residual=np.abs(depth_render[valid]-depth_real[valid])
        # 平均深度误差
        x1_real=np.mean(residual)*100
    else:
        # 完全没有匹配区域
        x1_real=20.0
    return x1_real

def reliability_inlier_ratio(valid_depth, Z_pred, Z_real):
    if valid_depth.sum()>0:
            Z_pred_valid = Z_pred[valid_depth]
            Z_real_valid = Z_real[valid_depth]

            depth_diff = np.abs(Z_pred_valid-Z_real_valid)*100
            x3_score = 1 - np.mean(depth_diff < 2.0)     
    else:
        x3_score = 1

    return x3_score


def main(args):
    csv_rows = []

    renders_obj = []
    d_objs = []  # 初始化物体 3D 直径 (cm)
    scenes = []
    mesh_nodes = []
    models_pts=[]
    model_names = []
    for model_seq in args.cad_models_seq:
        points_path = os.path.join(args.mesh_path_root, model_seq, f"points.xyz")
        mesh_path = os.path.join(args.mesh_path_root, model_seq, f"textured.obj")
        mesh = trimesh.load(mesh_path)
        render_mesh = pyrender.Mesh.from_trimesh(mesh,smooth=False)
        scene = pyrender.Scene()

        mesh_node = scene.add(render_mesh)
        mesh_nodes.append(mesh_node)

        camera = pyrender.IntrinsicsCamera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
        camera_node = scene.add(camera, pose=np.eye(4)) 
        scenes.append(scene)

        renderer = pyrender.OffscreenRenderer(viewport_width=640,viewport_height=480)
        renders_obj.append(renderer)

        with open(points_path, 'r') as f:
            model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
            models_pts.append(model_pts)
            
        open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))
        open3d_models.append(open3d_model)

        # 2.【2 行代码动态计算 3D 直径 d_obj (cm)】
        bbox_min = np.min(model_pts, axis=0) # 找到 [x_min, y_min, z_min]
        bbox_max = np.max(model_pts, axis=0) # 找到 [x_max, y_max, z_max]
        # 计算 3D 包围盒对角线长度 (m -> cm)
        d = np.linalg.norm(bbox_max - bbox_min) * 100  
        d_objs.append(d)
        print(f" 物体 3D 直径 d_obj = {d:.2f} cm")

    obj_idx = 0
    for seq_target in args.target_seqs:

        # print(obj_idx)
        # print(len(renders_obj), renders_obj[obj_idx])
        # print(len(mesh_nodes),  mesh_nodes[obj_idx])
        # print(len(open3d_models),open3d_models[obj_idx])

        for dot in args.corruption_lists:   
            seq = seq_target + dot
            pred_files = sorted(glob.glob(f"{args.res_dir}/{seq_target}/{seq}/*.txt"))
            raw_seq_name = seq_target
            gt_files = sorted(glob.glob(f"{args.ycb_dir}/{raw_seq_name}/annotated_poses/*.txt"))
            depth_files = sorted(glob.glob(f"{args.data_dir}/{seq}/depth/*.png"))
            # print(f"{res_dir}/{seq_target}/{seq}/*.txt")

            for i in range(min(len(pred_files), len(gt_files))):
                if i < 2:
                    T_prior=T_obs = np.loadtxt(pred_files[i])
                    T_obs = np.loadtxt(pred_files[i])
                    T_gt = np.loadtxt(gt_files[i])
                #print(f"\n正在处理受损序列: {pred_files[i]} ...")
                else:
                    T_obs = np.loadtxt(pred_files[i])
                    T_gt = np.loadtxt(gt_files[i])
                    T_prev1 = np.loadtxt(pred_files[i-1])
                    T_prev2 = np.loadtxt(pred_files[i-2])
                    # A. 计算 SE(3) 恒定速度惯性先验 T_prior
                    delta1 = se3_log_map(np.linalg.inv(T_prev2) @ T_prev1)
                    T_prior = T_prev1 @ se3_exp_map(delta1)
                

                # B. 计算两者的绝对 ADD-S 姿态误差 (cm)
                E_update_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100
                E_prior_cm = U.adi(T_prior, T_gt, open3d_models[obj_idx]) * 100

                # C. 【物体系数归一化】: 除以物体 3D 直径 d_obj
                e_update_norm = E_update_cm / d_objs[obj_idx]
                e_prior_norm = E_prior_cm / d_objs[obj_idx]
                delta_norml = args.delta / d_objs[obj_idx]  # 归一化的阈值 
                
                # 🌟 D. 核心help 标签计算：判断听视觉是否输给了听惯性！
                help_label = 1 if ((e_update_norm + delta_norml) < e_prior_norm ) else 0
                # E. 提取 4 个完全部署级的特征 (无 GT 依赖)
                depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
                depth_real = depth_raw.astype(np.float32) / 1000.0
                
                x1_depth_residual = reliability_depth_residual(depth_real, T_obs, scenes[obj_idx],renders_obj[obj_idx],mesh_nodes[obj_idx])


                R_p, t_p = T_obs[:3, :3], T_obs[:3, 3]
                pts_cam = (R_p @ models_pts[obj_idx].T).T + t_p
                X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
                u = np.round((K[0, 0] * X / Z) + K[0, 2]).astype(int)
                v = np.round((K[1, 1] * Y / Z) + K[1, 2]).astype(int)
                valid_bounds = (u >= 0) & (u < 640) & (v >= 0) & (v < 480) & (Z > 0)
                u_v, v_v, Z_p= u[valid_bounds], v[valid_bounds], Z[valid_bounds]
                projected_mask = np.zeros((480,640), dtype=np.uint8)
                projected_mask[v_v, u_v] = 1
                Z_real = depth_real[v_v, u_v]
                valid_depth = Z_real > 0
                x2_inlier_ratio = reliability_inlier_ratio(valid_depth, Z_p, Z_real)

                innovation_vec = se3_log_map(np.linalg.inv(T_prior) @ T_obs)
                x3_innovation_mag = np.linalg.norm(innovation_vec) 
                x3_trans_innovation = np.linalg.norm(innovation_vec[:3])        
                x3_rot_innovation = np.linalg.norm(innovation_vec[3:])  # 时序新息模长

                if len(u_v) > 0:                    
                    x4_support_ratio = 1-(np.sum(Z_real > 0.1) / (len(Z_real) + 1e-5)) # 有效支撑率
                else:
                    x4_support_ratio = 10.0, 0.0

                # 保存为 CSV 的一行记录
                csv_rows.append({
                    "sequence": seq,
                    "frame_idx": i,
                    "E_update_cm": E_update_cm,
                    "E_prior_cm": E_prior_cm,
                    "e_update_norm": e_update_norm,
                    "e_prior_norm": e_prior_norm,
                    "help_label": help_label,                  
                    "x1_depth_residual": x1_depth_residual,
                    "x2_inlier_ratio": x2_inlier_ratio,
                    "x3_innovation_mag": x3_innovation_mag,
                    "x3_trans_innovation": x3_trans_innovation,
                    "x3_rot_innovation": x3_rot_innovation,
                    "x4_support_ratio": x4_support_ratio,
                    "D_obj":d_objs[obj_idx]
                })
        obj_idx += 1  # 切换到下一个物体的 3D 直径

    # 导出为表格文件
    df = pd.DataFrame(csv_rows)
    df.to_csv(f"./per_frame_help_dataset_delta{args.delta}.csv", index=False)

    print("\n" + "="*50)
    print("✅ 逐帧 CSV 标签数据集导出成功: ./per_frame_help_dataset.csv！")
    print(f"数据总行数: {len(df)} 行 | Help=1 (视觉有益帧) 占比: {df['help_label'].mean()*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 YCBInEOAT Help标签")
    parser.add_argument('--ycb_dir', type=str, default="./datasets/YCBInEOAT", help="原始数据集基础路径")
    parser.add_argument('--data_dir', type=str, default="./datasets/YCBInEOAT_Corrupted", help="受损数据集基础路径")
    parser.add_argument('--res_dir', type=str, default="./results_collection", help="SE(3)TrackNet预测的位姿保存路径")
    parser.add_argument('--mesh_path_root', type=str, default="./datasets/YCB_Video_Models/CADmodels", help="CAD模型保存路径")
    parser.add_argument('--target_seqs', nargs='+', default=["mustard0", "bleach_hard_00_03_chaitanya", "bleach0"], help="要处理的序列名称列表")
    parser.add_argument('--corruption_lists', nargs='+', default=["_occ40","_black10","_clean","_drop60","_occ60"], help="corruption_lists")
    parser.add_argument('--cad_models_seq', nargs='+', default=["006_mustard_bottle","021_bleach_cleanser","021_bleach_cleanser"], help="CAD model list")
    parser.add_argument('--delta', type=float, default= 0.2, help="marigin_rate")
    args = parser.parse_args()
    main(args)