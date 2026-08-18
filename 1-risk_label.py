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
import open3d  as o3d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler



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
def build_safe_depth_dict(depth_folder, gt_dict):
    """
    将时间戳命名的 depth 文件安全映射到整数 frame_id (0, 1, 2...)
    加入时间戳严格单调递增校验，彻底消灭错位风险！
    """
    # 1. 获取所有以时间戳命名的深度图文件
    depth_files = glob.glob(os.path.join(depth_folder, "*.png"))
    # 2. 从文件名提取纯数字时间戳，并按时间戳升序严格排序
    def extract_timestamp(path):
        digits = ''.join(filter(str.isdigit, os.path.basename(path)))
        return int(digits) if len(digits) > 0 else 0
        
    depth_files_sorted = sorted(depth_files, key=extract_timestamp)
    gt_frame_ids = sorted(gt_dict.keys()) # [0, 1, 2, ..., N]

    # 3. 严格长度与非空断言 (防丢帧)
    assert len(depth_files_sorted) == len(gt_frame_ids), \
        f"错误: 深度图数量 ({len(depth_files_sorted)}) 与 GT 帧数 ({len(gt_frame_ids)}) 不一致！"

    # 4. 校验时间戳是否严格单调递增 (防乱序)
    timestamps = [extract_timestamp(f) for f in depth_files_sorted]
    assert all(timestamps[k] < timestamps[k+1] for k in range(len(timestamps)-1)), \
        "错误: 深度图时间戳非严格单调递增！存在乱序帧！"

    # 5. 建立以整数 frame_id 为 Key 的安全字典
    depth_dict = {}
    for k, frame_id in enumerate(gt_frame_ids):
        depth_dict[frame_id] = depth_files_sorted[k]

    return depth_dict

def actual_recovery_action(current_depth_real, T_obs, model_pts_3d, K):  

    model_h=np.hstack([model_pts_3d,np.ones((len(model_pts_3d),1))])
    model_cam=(T_obs @ model_h.T).T[:,:3]
    z_min=model_cam[:,2].min()
    z_max=model_cam[:,2].max()
    
    margin=0.05
    mask=((current_depth_real>z_min-margin)&(current_depth_real<z_max+margin))
    y_idx,x_idx=np.where(mask)

    if len(y_idx) < 100:
        print("recovery的输出为T_prior")
        T = False
        return 0, False # 依然全黑，无法恢复，继续听先验
    
    z_vals = current_depth_real[y_idx, x_idx]
    x_c = (x_idx - K[0, 2]) * z_vals / K[0, 0]
    y_c = (y_idx - K[1, 2]) * z_vals / K[1, 1]
    camera_pts = np.vstack((x_c, y_c, z_vals)).T
    
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(camera_pts)
    
    # 2. 准备物体的 CAD 点云
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(model_pts_3d)
    
    # 以 T_prior 为初始猜测，但在 5cm 范围内自由寻找最佳吻合位置
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_obj, pcd_cam, max_correspondence_distance=0.05,
        init=T_obs,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    T = True
    print("recovery的输出为T_icp")
    return icp_result.transformation,T



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

    print("阶段 1: 提取单帧观测特征并训练初级观察模型 clf_obs...")
    stage1_X_obs = []
    stage1_y_obs = []
    
    obj_idx = 0
    for seq_target in args.target_seqs:
        for dot in args.corruption_lists:              
            seq = seq_target + dot            
            pred_dict = build_frame_dict(f"{args.res_dir}/{seq_target}/{seq}")
            gt_dict   = build_frame_dict(f"{args.ycb_dir}/{seq_target}/annotated_poses")
            depth_dict = build_safe_depth_dict(f"{args.data_dir}/{seq}/depth", gt_dict)
            matched_frames = sorted(set(pred_dict.keys()) & set(gt_dict.keys()) & set(depth_dict.keys()))

            for frame_id in matched_frames:
                T_obs = np.loadtxt(pred_dict[frame_id]).reshape(4, 4)
                T_gt  = np.loadtxt(gt_dict[frame_id]).reshape(4, 4)
                depth_raw = cv2.imread(depth_dict[frame_id], cv2.IMREAD_UNCHANGED)
                depth_real = depth_raw.astype(np.float32) / 1000.0

                # 计算观察误差与观察标签
                E_update_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100
                e_update_norm = E_update_cm / d_objs[obj_idx]
                risk_threshold_norml = args.risk_threshold / d_objs[obj_idx]
                obs_risk_label = int(e_update_norm > risk_threshold_norml)

                # 提取单帧特征 x1, x4
                x1_depth_res = reliability_depth_residual(depth_real, T_obs, scenes[obj_idx], renders_obj[obj_idx], mesh_nodes[obj_idx])
                
                # 快速支撑率计算
                R_p, t_p = T_obs[:3, :3], T_obs[:3, 3]
                pts_cam = (R_p @ models_pts[obj_idx].T).T + t_p
                u = np.round((K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2]) + K[0, 2]).astype(int)
                v = np.round((K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2]) + K[1, 2]).astype(int)
                valid = (u >= 0) & (u < 640) & (v >= 0) & (v < 480) & (pts_cam[:, 2] > 0)
                Z_real = depth_real[v[valid], u[valid]]
                x4_support = 1.0 - (np.sum(Z_real > 0.1) / (len(Z_real) + 1e-5)) if len(Z_real) > 0 else 1.0

                stage1_X_obs.append([x1_depth_res, x4_support])
                stage1_y_obs.append(obs_risk_label)
        obj_idx += 1

    # 训练第一阶段的 clf_obs (只看单帧特征，完全独立于时序历史!)
    scaler_obs = MinMaxScaler()
    X_obs_scaled = scaler_obs.fit_transform(np.array(stage1_X_obs))
    clf_obs_stage1 = LogisticRegression().fit(X_obs_scaled, np.array(stage1_y_obs))
    print("✅ 阶段 1 完成: 成功训练纯单帧观察分类器 clf_obs_stage1！\n")


    print("阶段 2: 让 B5 自主闭环滚推 (零真值干预)，生成真正的 prior_risk_label 并导出 CSV...")
    csv_rows = []
    obj_idx = 0

    for seq_target in args.target_seqs:
        for dot in args.corruption_lists:              
            seq = seq_target + dot            
            pred_dict = build_frame_dict(f"{args.res_dir}/{seq_target}/{seq}")
            gt_dict   = build_frame_dict(f"{args.ycb_dir}/{seq_target}/annotated_poses")
            depth_dict = build_safe_depth_dict(f"{args.data_dir}/{seq}/depth", gt_dict)
            matched_frames = sorted(set(pred_dict.keys()) & set(gt_dict.keys()) & set(depth_dict.keys()))

            T_B5_history = []
            history_frames = []

            for frame_id in matched_frames:
                T_obs = np.loadtxt(pred_dict[frame_id]).reshape(4, 4)
                T_gt  = np.loadtxt(gt_dict[frame_id]).reshape(4, 4)
                depth_raw = cv2.imread(depth_dict[frame_id], cv2.IMREAD_UNCHANGED)
                depth_real = depth_raw.astype(np.float32) / 1000.0

                # 1. 提取所有特征
                x1_depth_residual = reliability_depth_residual(depth_real, T_obs, scenes[obj_idx], renders_obj[obj_idx], mesh_nodes[obj_idx])
                
                R_p, t_p = T_obs[:3, :3], T_obs[:3, 3]
                pts_cam = (R_p @ models_pts[obj_idx].T).T + t_p
                X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
                u = np.round((K[0, 0] * X / Z) + K[0, 2]).astype(int)
                v = np.round((K[1, 1] * Y / Z) + K[1, 2]).astype(int)
                valid_bounds = (u >= 0) & (u < 640) & (v >= 0) & (v < 480) & (Z > 0)
                u_v, v_v, Z_p = u[valid_bounds], v[valid_bounds], Z[valid_bounds]
                Z_real = depth_real[v_v, u_v]
                
                x2_inlier_ratio = reliability_inlier_ratio(Z_real > 0, Z_p, Z_real)
                x4_support_ratio = 1.0 - (np.sum(Z_real > 0.1) / (len(Z_real) + 1e-5)) if len(Z_real) > 0 else 1.0

                #  2. 闭环先验 T_prior 完全来自于 B5 过去的自主历史 T_B5_history！
                if len(T_B5_history) < 2:
                    T_prior = T_obs
                    T_final_B5 = T_obs
                else:
                    T_prev1 = T_B5_history[-1]
                    T_prev2 = T_B5_history[-2]
                    delta1 = se3_log_map(np.linalg.inv(T_prev2) @ T_prev1)
                    T_prior = (T_prev1 @ se3_exp_map(delta1))

                    # 3. B5 自身做决策 (使用阶段 1 训练好的模型预测，绝对不看真值 T_gt 做分支选择！)
                    feat_scaled = scaler_obs.transform([[x1_depth_residual, x4_support_ratio]])
                    p_obs_bad = clf_obs_stage1.predict_proba(feat_scaled)[0, 1]

                    if p_obs_bad < 0.80:
                        T_final_B5 = T_obs       # B5 决定听视觉
                    else:
                        T_final_B5 = T_prior     # B5 决定听惯性

                T_B5_history.append(T_final_B5)
                history_frames.append(frame_id)

                # 4. 计算时序新息特征 x3
                innovation_vec = se3_log_map(np.linalg.inv(T_prior) @ T_obs)
                x3_innovation_mag = np.linalg.norm(innovation_vec)
                x3_trans_innovation = np.linalg.norm(innovation_vec[:3])
                x3_rot_innovation = np.linalg.norm(innovation_vec[3:])

                # 5. 真实打标签：评估两者误差
                E_update_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100
                E_prior_cm  = U.adi(T_prior, T_gt, open3d_models[obj_idx]) * 100

                e_update_norm = E_update_cm / d_objs[obj_idx]
                e_prior_norm  = E_prior_cm / d_objs[obj_idx]
                risk_threshold_norml = args.risk_threshold / d_objs[obj_idx]

                obs_risk_label   = int(e_update_norm > risk_threshold_norml)
                prior_risk_label = int(e_prior_norm > risk_threshold_norml)

                csv_rows.append({
                    "sequence": seq,
                    "frame_id": frame_id,
                    "E_update_cm": E_update_cm,
                    "E_prior_cm": E_prior_cm,
                    "e_update_norm": e_update_norm,
                    "e_prior_norm": e_prior_norm,
                    "obs_risk_label": obs_risk_label,
                    "prior_risk_label": prior_risk_label, 
                    "x1_depth_residual": x1_depth_residual,
                    "x2_inlier_ratio": x2_inlier_ratio,
                    "x3_innovation_mag": x3_innovation_mag,
                    "x3_trans_innovation": x3_trans_innovation,
                    "x3_rot_innovation": x3_rot_innovation,
                    "x4_support_ratio": x4_support_ratio,
                    "D_obj": d_objs[obj_idx]
                })
        obj_idx += 1

    for episode in args.ci_episode:

            seq = args.ci_object + episode           
            pred_dict = build_frame_dict(f"{args.res_dir}/{args.ci_object}/{seq}")
            raw_seq_name = args.ci_object
            gt_dict   = build_frame_dict(f"{args.ycb_dir}/{raw_seq_name}/annotated_poses")
            depth_dict={}
            depth_dict = build_safe_depth_dict(f"{args.data_dir}/{seq}/depth", gt_dict)
            matched_frames = sorted(set(pred_dict.keys()) & set(gt_dict.keys()) & set(depth_dict.keys()))

            T_B5_history = []
            history_frames = []
            for frame_id in matched_frames:
                T_obs = np.loadtxt(pred_dict[frame_id]).reshape(4, 4)
                T_gt  = np.loadtxt(gt_dict[frame_id]).reshape(4, 4)
                depth_raw = cv2.imread(depth_dict[frame_id], cv2.IMREAD_UNCHANGED)
                depth_real = depth_raw.astype(np.float32) / 1000.0
                # 1. 提取所有特征
                x1_depth_residual = reliability_depth_residual(depth_real, T_obs, scenes[obj_idx], renders_obj[obj_idx], mesh_nodes[obj_idx])
                
                R_p, t_p = T_obs[:3, :3], T_obs[:3, 3]
                pts_cam = (R_p @ models_pts[obj_idx].T).T + t_p
                X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
                u = np.round((K[0, 0] * X / Z) + K[0, 2]).astype(int)
                v = np.round((K[1, 1] * Y / Z) + K[1, 2]).astype(int)
                valid_bounds = (u >= 0) & (u < 640) & (v >= 0) & (v < 480) & (Z > 0)
                u_v, v_v, Z_p = u[valid_bounds], v[valid_bounds], Z[valid_bounds]
                Z_real = depth_real[v_v, u_v]
                
                x2_inlier_ratio = reliability_inlier_ratio(Z_real > 0, Z_p, Z_real)
                x4_support_ratio = 1.0 - (np.sum(Z_real > 0.1) / (len(Z_real) + 1e-5)) if len(Z_real) > 0 else 1.0

                #  2. 闭环先验 T_prior 完全来自于 B5 过去的自主历史 T_B5_history！
                if len(T_B5_history) < 2:
                    T_prior = T_obs
                    T_final_B5 = T_obs
                else:
                    T_prev1 = T_B5_history[-1]
                    T_prev2 = T_B5_history[-2]
                    delta1 = se3_log_map(np.linalg.inv(T_prev2) @ T_prev1)
                    T_prior = (T_prev1 @ se3_exp_map(delta1))

                    # 3. B5 自身做决策 (使用阶段 1 训练好的模型预测，绝对不看真值 T_gt 做分支选择！)
                    feat_scaled = scaler_obs.transform([[x1_depth_residual, x4_support_ratio]])
                    p_obs_bad = clf_obs_stage1.predict_proba(feat_scaled)[0, 1]

                    if p_obs_bad < 0.80:
                        T_final_B5 = T_obs       # B5 决定听视觉
                    else:
                        T_final_B5 = T_prior     # B5 决定听惯性

                T_B5_history.append(T_final_B5)
                history_frames.append(frame_id)

                # 4. 计算时序新息特征 x3
                innovation_vec = se3_log_map(np.linalg.inv(T_prior) @ T_obs)
                x3_innovation_mag = np.linalg.norm(innovation_vec)
                x3_trans_innovation = np.linalg.norm(innovation_vec[:3])
                x3_rot_innovation = np.linalg.norm(innovation_vec[3:])

                # 5. 真实打标签：评估两者误差
                E_update_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100
                E_prior_cm  = U.adi(T_prior, T_gt, open3d_models[obj_idx]) * 100

                e_update_norm = E_update_cm / d_objs[obj_idx]
                e_prior_norm  = E_prior_cm / d_objs[obj_idx]
                risk_threshold_norml = args.risk_threshold / d_objs[obj_idx]

                obs_risk_label   = int(e_update_norm > risk_threshold_norml)
                prior_risk_label = int(e_prior_norm > risk_threshold_norml)

                csv_rows.append({
                    "sequence": seq,
                    "frame_id": frame_id,
                    "E_update_cm": E_update_cm,
                    "E_prior_cm": E_prior_cm,
                    "e_update_norm": e_update_norm,
                    "e_prior_norm": e_prior_norm,
                    "obs_risk_label": obs_risk_label,
                    "prior_risk_label": prior_risk_label, 
                    "x1_depth_residual": x1_depth_residual,
                    "x2_inlier_ratio": x2_inlier_ratio,
                    "x3_innovation_mag": x3_innovation_mag,
                    "x3_trans_innovation": x3_trans_innovation,
                    "x3_rot_innovation": x3_rot_innovation,
                    "x4_support_ratio": x4_support_ratio,
                    "D_obj": d_objs[obj_idx]
                })

    # 导出为表格文件
    df = pd.DataFrame(csv_rows)
    df.to_csv(f"./per_frame_label_threshold{args.risk_threshold}.csv", index=False)

    balance_df = df.groupby('sequence').agg(
        Total_Frames=('obs_risk_label', 'count'),
        Obs_Risk_Positive_Ratio=('obs_risk_label', lambda x: f"{x.mean()*100:.2f}%"),
        Prior_Risk_Positive_Ratio=('prior_risk_label', lambda x: f"{x.mean()*100:.2f}%")
    ).reset_index()

    balance_csv_path = f"./class_balance_summary_threshold{args.risk_threshold}.csv"
    balance_df.to_csv(balance_csv_path, index=False)


    print("\n" + "="*50)
    print(f"数据总行数: {len(df)} 行 | obs_risk=1 占比: {df['obs_risk_label'].mean()*100:.2f}%")
    print(f"数据总行数: {len(df)} 行 | prior_risk=1 占比: {df['prior_risk_label'].mean()*100:.2f}%")
    print(f"✅ 逐帧日志已保存至: ./per_frame_label_threshold{args.risk_threshold}.csv")
    print(f"✅ 类别平衡汇总表已保存至: {balance_csv_path}")
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