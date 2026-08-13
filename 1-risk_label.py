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


def build_frame_dict(folder):

    frame_dict={}
    files=glob.glob(os.path.join(folder,"*.txt"))
    for f in files:
        frame_id=int(os.path.splitext(os.path.basename(f))[0])
        frame_dict[frame_id]=f

    return frame_dict


def build_depth_dict(depth_dir, gt_dir):

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

    gt_files = sorted(
        glob.glob(os.path.join(gt_dir, "*.txt"),key=lambda x:int(os.path.splitext(os.path.basename(x))[0])))
    assert len(depth_files)==len(gt_files)

    depth_dict={}

    for depth_file, gt_file in zip(depth_files, gt_files):

        frame_id = int(os.path.splitext(os.path.basename(gt_file))[0])
        depth_dict[frame_id]=depth_file

    return depth_dict



def main(args):
    csv_rows = []
    renders_obj = []
    d_objs = []  # 初始化物体 3D 直径 (cm)
    scenes = []
    mesh_nodes = []
    models_pts=[]
    open3d_models = []

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
        for dot in args.corruption_lists:              
            seq = seq_target + dot            
            pred_dict = build_frame_dict(f"{args.res_dir}/{seq_target}/{seq}")
            raw_seq_name = seq_target
            gt_dict = build_frame_dict(f"{args.ycb_dir}/{raw_seq_name}/annotated_poses")
            matched_frames = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))

            depth_files = sorted(glob.glob(f"{args.data_dir}/{seq}/depth/*.png"))
            frame_ids = sorted(gt_dict.keys())
            depth_dict={}
            
            for frame_id, depth_path in zip(frame_ids, depth_files):
                depth_dict[frame_id]=depth_path

            # print(f"{res_dir}/{seq_target}/{seq}/*.txt")
            history_frames = []
            history_priors = {}
            for frame_id in matched_frames:
                T_obs=np.loadtxt(pred_dict[frame_id])
                T_gt=np.loadtxt(gt_dict[frame_id])
                
                if len(history_frames)<2: 
                    T_prior=T_obs     
                         
                else:
                    T_prev1= history_priors[history_frames[-1]]
                    T_prev2= history_priors[history_frames[-2]]
                    delta1 = se3_log_map(np.linalg.inv(T_prev2) @ T_prev1)
                    T_prior = (T_prev1 @ se3_exp_map(delta1))
                history_frames.append(frame_id)  
                
                # B. 计算两者的绝对 ADD-S 姿态误差 (cm)
                E_update_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100
                E_prior_cm = U.adi(T_prior, T_gt, open3d_models[obj_idx]) * 100

                # C. 【物体系数归一化】: 除以物体 3D 直径 d_obj
                e_update_norm = E_update_cm / d_objs[obj_idx]
                e_prior_norm = E_prior_cm / d_objs[obj_idx]
                risk_threshold_norml = args.risk_threshold / d_objs[obj_idx]  # 归一化的阈值 

                obs_risk_label = int(e_update_norm > risk_threshold_norml)
                prior_risk_label = int(e_prior_norm > risk_threshold_norml)
                history_priors[frame_id] = T_prior
                
            
                # E. 提取 4 个完全部署级的特征 (无 GT 依赖)
                depth_raw = cv2.imread(depth_dict[frame_id], cv2.IMREAD_UNCHANGED)
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
                    "frame_id": frame_id,
                    "E_update_cm": E_update_cm,
                    "E_prior_cm": E_prior_cm,
                    "e_update_norm": e_update_norm,
                    "e_prior_norm": e_prior_norm,
                    "obs_risk_label":obs_risk_label,
                    "prior_risk_label":prior_risk_label,   
                    "x1_depth_residual": x1_depth_residual,
                    "x2_inlier_ratio": x2_inlier_ratio,
                    "x3_innovation_mag": x3_innovation_mag,
                    "x3_trans_innovation": x3_trans_innovation,
                    "x3_rot_innovation": x3_rot_innovation,
                    "x4_support_ratio": x4_support_ratio,
                    "D_obj":d_objs[obj_idx]
                })
        obj_idx += 1  # 切换到下一个物体的 3D 直径


    for episode in args.ci_episode:
        seq = args.ci_object + episode
        pred_dict = build_frame_dict(f"{args.res_dir}/{args.ci_object}/{seq}")
        raw_seq_name = args.ci_object
        gt_dict = build_frame_dict(f"{args.ycb_dir}/{raw_seq_name}/annotated_poses")
        matched_frames = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))

        depth_files = sorted(glob.glob(f"{args.data_dir}/{seq}/depth/*.png"))
        frame_ids = sorted(gt_dict.keys())
        depth_dict={}
        

        for frame_id, depth_path in zip(frame_ids, depth_files):
            depth_dict[frame_id]=depth_path

        # print(f"{res_dir}/{seq_target}/{seq}/*.txt")
        history_frames = []
        history_priors = {}
        for frame_id in matched_frames:
            T_obs=np.loadtxt(pred_dict[frame_id])
            T_gt=np.loadtxt(gt_dict[frame_id])
            
            if len(history_frames)<2: 
                T_prior=T_obs     
                        
            else:
                T_prev1=history_priors[history_frames[-1]]
                T_prev2=history_priors[history_frames[-2]]
                delta1 = se3_log_map(np.linalg.inv(T_prev2) @ T_prev1)
                T_prior = (T_prev1 @ se3_exp_map(delta1))
            history_frames.append(frame_id)  
            # B. 计算两者的绝对 ADD-S 姿态误差 (cm)
            E_update_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100
            E_prior_cm = U.adi(T_prior, T_gt, open3d_models[obj_idx]) * 100

            # C. 【物体系数归一化】: 除以物体 3D 直径 d_obj
            e_update_norm = E_update_cm / d_objs[obj_idx]
            e_prior_norm = E_prior_cm / d_objs[obj_idx]
            risk_threshold_norml = args.risk_threshold / d_objs[obj_idx]  # 归一化的阈值 
            
            obs_risk_label = int(e_update_norm > risk_threshold_norml)
            prior_risk_label = int(e_prior_norm > risk_threshold_norml)
            history_priors[frame_id] = T_prior
            
            # E. 提取 4 个完全部署级的特征 (无 GT 依赖)
            depth_raw = cv2.imread(depth_dict[frame_id], cv2.IMREAD_UNCHANGED)
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
                "frame_id": frame_id,
                "E_update_cm": E_update_cm,
                "E_prior_cm": E_prior_cm,
                "e_update_norm": e_update_norm,
                "e_prior_norm": e_prior_norm,
                "obs_risk_label":obs_risk_label,
                "prior_risk_label":prior_risk_label,                  
                "x1_depth_residual": x1_depth_residual,
                "x2_inlier_ratio": x2_inlier_ratio,
                "x3_innovation_mag": x3_innovation_mag,
                "x3_trans_innovation": x3_trans_innovation,
                "x3_rot_innovation": x3_rot_innovation,
                "x4_support_ratio": x4_support_ratio,
                "D_obj":d_objs[obj_idx]
            })


    # 导出为表格文件
    df = pd.DataFrame(csv_rows)
    df.to_csv(f"./per_frame_label_threshold{args.risk_threshold}.csv", index=False)

    print("\n" + "="*50)
    print("✅ 逐帧 CSV 标签数据集导出成功: ./per_frame_help_dataset.csv！")
    print(f"数据总行数: {len(df)} 行 | obs_risk=1 占比: {df['obs_risk_label'].mean()*100:.2f}%")
    print(f"数据总行数: {len(df)} 行 | prior_risk=1 占比: {df['prior_risk_label'].mean()*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 YCBInEOAT Help标签")
    parser.add_argument('--ycb_dir', type=str, default="./datasets/YCBInEOAT", help="原始数据集基础路径")
    parser.add_argument('--data_dir', type=str, default="./datasets/YCBInEOAT_Corrupted", help="受损数据集基础路径")
    parser.add_argument('--res_dir', type=str, default="./results_collection", help="SE(3)TrackNet预测的位姿保存路径")
    parser.add_argument('--mesh_path_root', type=str, default="./datasets/YCB_Video_Models/CADmodels", help="CAD模型保存路径")
    parser.add_argument('--target_seqs', nargs='+', default=["mustard0", "bleach_hard_00_03_chaitanya", "bleach0"], help="要处理的序列名称列表")
    parser.add_argument('--corruption_lists', nargs='+', default=["_occ40","_black10","_clean","_drop60","_occ60"], help="corruption_lists")  #
    parser.add_argument('--ci_object', type=str, default="bleach_hard_00_03_chaitanya", help="object used for computing CI")
    parser.add_argument('--ci_episode',nargs='+', type=str, default=["_black10_2","_black10_3","_black10_4","_black10_5"], help="condition used for computing CI")
    parser.add_argument('--cad_models_seq', nargs='+', default=["006_mustard_bottle","021_bleach_cleanser","021_bleach_cleanser","021_bleach_cleanser"], help="CAD model list")  
    parser.add_argument('--risk_threshold', type=float, default= 0.5, help="marigin_rate")
    args = parser.parse_args()
    main(args)