import os
import numpy as np
import pandas as pd
import cv2
import Utils as U
from scipy.spatial.transform import Rotation as R_sci
import trimesh
import pyrender
import argparse
import open3d as o3d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

K = np.array([
    [3.195820007324218750e+02, 0.0, 3.202149847676955687e+02],
    [0.0, 4.171186828613281250e+02, 2.443486680871046701e+02],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

cv_to_gl = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def se3_log_map(T):
    R_mat = T[:3, :3]
    t_vec = T[:3, 3]
    w_vec = R_sci.from_matrix(R_mat).as_rotvec()
    return np.concatenate([t_vec, w_vec])

def se3_exp_map(delta):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_sci.from_rotvec(delta[3:]).as_matrix()
    T[:3, 3] = delta[:3]
    return T

def reliability_depth_residual(depth_real, pred_pose, scene, renderer, mesh_node):
    pose_render = cv_to_gl @ pred_pose
    scene.set_pose(mesh_node, pose_render)
    depth_render = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    valid = depth_render > 0
    if np.sum(valid) > 20:
        residual = np.abs(depth_render[valid] - depth_real[valid])
        return np.mean(residual) * 100
    return 20.0

def reliability_inlier_ratio(valid_depth, Z_pred, Z_real):
    if valid_depth.sum() > 0:
        Z_pred_valid = Z_pred[valid_depth]
        Z_real_valid = Z_real[valid_depth]
        depth_diff = np.abs(Z_pred_valid - Z_real_valid) * 100
        return 1 - np.mean(depth_diff < 2.0)
    return 1.0

def actual_recovery_action(current_depth_real, model_pts_3d, K):
    y_idx, x_idx = np.where((current_depth_real > 0.5) & (current_depth_real < 1.2))
    if len(y_idx) < 300:
        print("[Recovery] 有效深度点过少，恢复失败")
        return None, False
    z = current_depth_real[y_idx, x_idx]
    x = (x_idx - K[0, 2]) * z / K[0, 0]
    y = (y_idx - K[1, 2]) * z / K[1, 1]
    scene_pts = np.vstack([x, y, z]).T
    pcd_scene = o3d.geometry.PointCloud()
    pcd_scene.points = o3d.utility.Vector3dVector(scene_pts)
    _, inliers = pcd_scene.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=500)
    pcd_objects = pcd_scene.select_by_index(inliers, invert=True)
    if len(pcd_objects.points) < 50:
        pcd_objects = pcd_scene
    obj_pts = np.asarray(pcd_objects.points)
    real_object_center = np.median(obj_pts, axis=0)
    model_center = np.mean(model_pts_3d, axis=0)
    T_coarse = np.eye(4)
    T_coarse[:3, 3] = real_object_center - model_center
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(model_pts_3d)
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_model, pcd_objects, max_correspondence_distance=0.10, init=T_coarse,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    if icp_result.fitness < 0.15:
        print("[Recovery] ICP recovery failed")
        return None, False
    return icp_result.transformation, True

def b5_transition(T_obs, T_prior, p_obs_bad, p_prior_bad, p_risk_threshold,
                  support, exited_blackout, depth_real, model_pts,
                  use_prior_predictor=True):

    if support == 1:
        return T_prior, "MODE_3_BLACKOUT_WAITING"
    if exited_blackout:
        T_recovery, ok = actual_recovery_action(depth_real, model_pts, K)
        if ok:
            return T_recovery, "MODE_3_RECOVERY_EXECUTE"
        return T_prior, "MODE_3_RECOVERY_FAILED_PRIOR"
    if p_obs_bad <= p_risk_threshold:
        return T_obs, "MODE_1_ACCEPT"
    if use_prior_predictor:
        if p_prior_bad <= p_risk_threshold:
            return T_prior, "MODE_2_PRIOR"
        T_delta = np.linalg.inv(T_prior) @ T_obs
        alpha = 0.5
        T_final = T_prior @ se3_exp_map(alpha * se3_log_map(T_delta))
        return T_final, "MODE_3_UNCERTAIN_FUSION"
    return T_prior, "MODE_BOOTSTRAP_PRIOR"

def resolve_path(path_value, root):
    path_value = str(path_value)
    if os.path.isabs(path_value):
        return path_value
    if os.path.exists(path_value):
        return path_value
    return os.path.normpath(os.path.join(root, path_value))

def get_episode_df(manifest, seq):
    episode_df = manifest[manifest["sequence"] == seq].copy()
    if len(episode_df) == 0:
        raise ValueError(f"Manifest中找不到序列: {seq}")
    episode_df["frame_id"] = episode_df["frame_id"].astype(int)
    episode_df = episode_df.sort_values("frame_id")
    if episode_df["frame_id"].duplicated().any():
        dup = episode_df.loc[episode_df["frame_id"].duplicated(), "frame_id"].tolist()
        raise ValueError(f"{seq} 存在重复frame_id: {dup}")
    return episode_df

def load_frame_from_manifest(row, args):
    pred_path = resolve_path(row["pred_path"], args.res_dir)
    gt_path = resolve_path(row["gt_path"], args.ycb_dir)
    depth_path = resolve_path(row["depth_path"], args.data_dir)
    T_obs = np.loadtxt(pred_path).reshape(4, 4)
    T_gt = np.loadtxt(gt_path).reshape(4, 4)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"无法读取Depth: {depth_path}")
    depth_real = depth_raw.astype(np.float32) / 1000.0
    return T_obs, T_gt, depth_real

def extract_frame_features(T_obs, depth_real, obj_idx, models_pts, scenes, renders_obj, mesh_nodes):
    x1_depth_residual = reliability_depth_residual(
        depth_real, T_obs, scenes[obj_idx], renders_obj[obj_idx], mesh_nodes[obj_idx]
    )
    R_p, t_p = T_obs[:3, :3], T_obs[:3, 3]
    pts_cam = (R_p @ models_pts[obj_idx].T).T + t_p
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid_z = Z > 1e-8
    u = np.zeros(len(Z), dtype=int)
    v = np.zeros(len(Z), dtype=int)
    u[valid_z] = np.round((K[0, 0] * X[valid_z] / Z[valid_z]) + K[0, 2]).astype(int)
    v[valid_z] = np.round((K[1, 1] * Y[valid_z] / Z[valid_z]) + K[1, 2]).astype(int)
    valid_bounds = valid_z & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    u_v, v_v, Z_p = u[valid_bounds], v[valid_bounds], Z[valid_bounds]
    Z_real = depth_real[v_v, u_v] if len(u_v) > 0 else np.array([], dtype=np.float32)
    x2_inlier_ratio = reliability_inlier_ratio(Z_real > 0, Z_p, Z_real) if len(Z_real) > 0 else 1.0
    x4_support_ratio = 1.0 - (np.sum(Z_real > 0.1) / (len(Z_real) + 1e-5)) if len(Z_real) > 0 else 1.0
    return x1_depth_residual, x2_inlier_ratio, x4_support_ratio

def build_label_row(seq, frame_id, T_obs, T_prior, T_gt, obj_idx, d_objs, open3d_models,
                    x1, x2, x3_mag, x3_trans, x3_rot, x4, p_obs_bad, p_prior_bad, mode):
    E_update_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100
    E_prior_cm = U.adi(T_prior, T_gt, open3d_models[obj_idx]) * 100
    e_update_norm = E_update_cm / d_objs[obj_idx]
    e_prior_norm = E_prior_cm / d_objs[obj_idx]
    return {
        "sequence": seq,
        "frame_id": frame_id,
        "E_update_cm": E_update_cm,
        "E_prior_cm": E_prior_cm,
        "e_update_norm": e_update_norm,
        "e_prior_norm": e_prior_norm,
        "obs_risk_label": int(E_update_cm > ARGS_RISK_THRESHOLD_CM),
        "prior_risk_label": int(E_prior_cm > ARGS_RISK_THRESHOLD_CM),
        "x1_depth_residual": x1,
        "x2_inlier_ratio": x2,
        "x3_innovation_mag": x3_mag,
        "x3_trans_innovation": x3_trans,
        "x3_rot_innovation": x3_rot,
        "x4_support_ratio": x4,
        "p_obs_bad_rollout": p_obs_bad,
        "p_prior_bad_rollout": p_prior_bad if p_prior_bad is not None else np.nan,
        "rollout_mode": mode,
        "D_obj": d_objs[obj_idx]
    }

def rollout_episode(episode_df, seq, obj_idx, args, models_pts, scenes, renders_obj, mesh_nodes,
                    d_objs, open3d_models, clf_obs, scaler_obs, clf_prior=None, scaler_prior=None,
                    use_prior_predictor=False):
    rows = []
    T_B5_history = []
    consecutive_blackout = 0
    exited_blackout = False
    steps_since_reset = 0
    feature_cols = ['x1_depth_residual', 'x2_inlier_ratio', 'x3_innovation_mag', 'x4_support_ratio']

    for _, row in episode_df.iterrows():
        frame_id = int(row["frame_id"])
        T_obs, T_gt, depth_real = load_frame_from_manifest(row, args)
        x1, x2, x4 = extract_frame_features(
            T_obs, depth_real, obj_idx, models_pts, scenes, renders_obj, mesh_nodes
        )

        # short-horizon reset：只在非blackout帧进行，避免黑屏中用坏观测reset
        if args.rollout_horizon > 0 and steps_since_reset >= args.rollout_horizon and x4 != 1:
            T_B5_history = []
            steps_since_reset = 0

        if len(T_B5_history) < 2:
            T_prior = T_obs
        else:
            T_prev1 = T_B5_history[-1]
            T_prev2 = T_B5_history[-2]
            delta1 = se3_log_map(np.linalg.inv(T_prev2) @ T_prev1)
            T_prior = T_prev1 @ se3_exp_map(delta1)

        innovation_vec = se3_log_map(np.linalg.inv(T_prior) @ T_obs)
        x3_mag = np.linalg.norm(innovation_vec)
        x3_trans = np.linalg.norm(innovation_vec[:3])
        x3_rot = np.linalg.norm(innovation_vec[3:])

        if clf_obs.n_features_in_ == 2:
            obs_feat = scaler_obs.transform([[x1, x4]])
        else:
            obs_feat = scaler_obs.transform([[x1, x2, x3_mag, x4]])
        p_obs_bad = clf_obs.predict_proba(obs_feat)[0, 1]

        p_prior_bad = None
        if use_prior_predictor:
            prior_feat = scaler_prior.transform([[x1, x2, x3_mag, x4]])
            p_prior_bad = clf_prior.predict_proba(prior_feat)[0, 1]

        if x4 == 1:
            consecutive_blackout += 1
        else:
            if consecutive_blackout >= args.blackout_min_frames:
                exited_blackout = True
            consecutive_blackout = 0

        T_final_B5, mode = b5_transition(
            T_obs=T_obs, T_prior=T_prior, p_obs_bad=p_obs_bad, p_prior_bad=p_prior_bad,
            p_risk_threshold=args.p_risk_threshold, support=x4, exited_blackout=exited_blackout,
            depth_real=depth_real, model_pts=models_pts[obj_idx],
            use_prior_predictor=use_prior_predictor
        )
        if exited_blackout and x4 != 1:
            exited_blackout = False

        T_B5_history.append(T_final_B5)
        steps_since_reset += 1

        rows.append(build_label_row(
            seq, frame_id, T_obs, T_prior, T_gt, obj_idx, d_objs, open3d_models,
            x1, x2, x3_mag, x3_trans, x3_rot, x4, p_obs_bad, p_prior_bad, mode
        ))
    return rows

def main(args):
    global ARGS_RISK_THRESHOLD_CM
    ARGS_RISK_THRESHOLD_CM = args.risk_threshold

    manifest = pd.read_csv(args.manifest_path)
    required_cols = {"base_sequence", "condition", "sequence", "frame_id", "depth_path", "gt_path", "pred_path"}
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"Manifest缺少字段: {sorted(missing_cols)}")
    if manifest.duplicated(["sequence", "frame_id"]).any():
        raise ValueError("Manifest中存在重复(sequence, frame_id)")

    renders_obj, d_objs, scenes, mesh_nodes, models_pts, open3d_models = [], [], [], [], [], []
    for model_seq in args.cad_models_seq[:len(args.target_seqs)]:
        points_path = os.path.join(args.mesh_path_root, model_seq, "points.xyz")
        mesh_path = os.path.join(args.mesh_path_root, model_seq, "textured.obj")
        mesh = trimesh.load(mesh_path)
        render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene()
        mesh_node = scene.add(render_mesh)
        mesh_nodes.append(mesh_node)
        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        scene.add(camera, pose=np.eye(4))
        scenes.append(scene)
        renders_obj.append(pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480))
        with open(points_path, 'r') as f:
            model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
        models_pts.append(model_pts)
        open3d_models.append(U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64)))
        bbox_min, bbox_max = np.min(model_pts, axis=0), np.max(model_pts, axis=0)
        d = np.linalg.norm(bbox_max - bbox_min) * 100
        d_objs.append(d)
        print(f"物体 3D 直径 d_obj = {d:.2f} cm")


    print("阶段 1: 训练 observation-risk warm-start predictor...")
    stage1_X_obs, stage1_y_obs = [], []
    for obj_idx, seq_target in enumerate(args.target_seqs):
        for dot in args.corruption_lists:
            seq = seq_target + dot
            episode_df = get_episode_df(manifest, seq)
            for _, row in episode_df.iterrows():
                T_obs, T_gt, depth_real = load_frame_from_manifest(row, args)
                x1, _, x4 = extract_frame_features(
                    T_obs, depth_real, obj_idx, models_pts, scenes, renders_obj, mesh_nodes
                )
                E_update_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100
                obs_risk_label = int(E_update_cm > args.risk_threshold)
                stage1_X_obs.append([x1, x4])
                stage1_y_obs.append(obs_risk_label)

    scaler_obs_warm = MinMaxScaler()
    X_obs_warm = scaler_obs_warm.fit_transform(np.asarray(stage1_X_obs))
    clf_obs_warm = LogisticRegression(max_iter=1000).fit(X_obs_warm, np.asarray(stage1_y_obs))
    print("阶段 1 完成。")


    print("阶段 2: bootstrap B5 rollout，生成初始prior-risk labels...")
    bootstrap_rows = []
    for obj_idx, seq_target in enumerate(args.target_seqs):
        for dot in args.corruption_lists:
            seq = seq_target + dot
            episode_df = get_episode_df(manifest, seq)
            bootstrap_rows += rollout_episode(
                episode_df, seq, obj_idx, args, models_pts, scenes, renders_obj, mesh_nodes,
                d_objs, open3d_models, clf_obs_warm, scaler_obs_warm,
                use_prior_predictor=False
            )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    feature_cols = ['x1_depth_residual', 'x2_inlier_ratio', 'x3_innovation_mag', 'x4_support_ratio']


    scaler_full = MinMaxScaler()
    X_boot = scaler_full.fit_transform(bootstrap_df[feature_cols].values)

    y_obs_boot = bootstrap_df['obs_risk_label'].values
    y_prior_boot = bootstrap_df['prior_risk_label'].values
    if len(np.unique(y_obs_boot)) < 2:
        raise ValueError("bootstrap obs_risk_label只有一个类别，无法训练obs predictor")
    if len(np.unique(y_prior_boot)) < 2:
        raise ValueError(
            "bootstrap prior_risk_label只有一个类别，无法训练prior predictor。"
            "请检查risk_threshold或使用short-horizon reset减小长期漂移。"
        )

    clf_obs_full = LogisticRegression(max_iter=1000).fit(X_boot, y_obs_boot)
    clf_prior_boot = LogisticRegression(max_iter=1000).fit(X_boot, y_prior_boot)
    print("阶段 2 完成:已得到用于最终label rollout的临时obs/prior predictor。")


    print("阶段 3: 使用完整 B5 transition 重新rollout并生成最终labels...")
    csv_rows = []

    for obj_idx, seq_target in enumerate(args.target_seqs):
        for dot in args.corruption_lists:
            seq = seq_target + dot
            episode_df = get_episode_df(manifest, seq)
            csv_rows += rollout_episode(
                episode_df, seq, obj_idx, args, models_pts, scenes, renders_obj, mesh_nodes,
                d_objs, open3d_models, clf_obs_full, scaler_full,
                clf_prior=clf_prior_boot, scaler_prior=scaler_full,
                use_prior_predictor=True
            )


    ci_obj_idx = args.target_seqs.index(args.ci_object)
    for episode in args.ci_episode:
        seq = args.ci_object + episode
        episode_df = get_episode_df(manifest, seq)
        csv_rows += rollout_episode(
            episode_df, seq, ci_obj_idx, args, models_pts, scenes, renders_obj, mesh_nodes,
            d_objs, open3d_models, clf_obs_full, scaler_full,
            clf_prior=clf_prior_boot, scaler_prior=scaler_full,
            use_prior_predictor=True
        )

    df = pd.DataFrame(csv_rows)
    output_csv = f"./per_frame_label_threshold{args.risk_threshold}.csv"
    df.to_csv(output_csv, index=False)

    balance_df = df.groupby('sequence').agg(
        Total_Frames=('obs_risk_label', 'count'),
        Obs_Risk_Positive_Ratio=('obs_risk_label', lambda x: f"{x.mean()*100:.2f}%"),
        Prior_Risk_Positive_Ratio=('prior_risk_label', lambda x: f"{x.mean()*100:.2f}%")
    ).reset_index()
    balance_csv_path = f"./class_balance_summary_threshold{args.risk_threshold}.csv"
    balance_df.to_csv(balance_csv_path, index=False)

    print("\n" + "=" * 60)
    print(f"数据总行数: {len(df)}")
    print(f"obs_risk=1 占比: {df['obs_risk_label'].mean()*100:.2f}%")
    print(f"prior_risk=1 占比: {df['prior_risk_label'].mean()*100:.2f}%")
    print(f"逐帧标签: {output_csv}")
    print(f"类别平衡: {balance_csv_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 YCBInEOAT observation/prior risk labels")
    parser.add_argument('--manifest_path', type=str, default="./dataset_manifest_all.csv")
    parser.add_argument('--ycb_dir', type=str, default="./datasets/YCBInEOAT", help="GT根目录")
    parser.add_argument('--data_dir', type=str, default="./datasets/YCBInEOAT_Corrupted", help="RGB/Depth根目录")
    parser.add_argument('--res_dir', type=str, default="./results_collection", help="Prediction根目录")
    parser.add_argument('--mesh_path_root', type=str, default="./datasets/YCB_Video_Models/CADmodels")
    parser.add_argument('--target_seqs', nargs='+', default=["mustard0", "bleach_hard_00_03_chaitanya", "bleach0"])
    parser.add_argument('--corruption_lists', nargs='+', default=["_occ40", "_black10", "_clean", "_drop60", "_occ60"])
    parser.add_argument('--ci_object', type=str, default="bleach_hard_00_03_chaitanya")
    parser.add_argument('--ci_episode', nargs='+', default=["_black10_2", "_black10_3", "_black10_4", "_black10_5"])
    parser.add_argument('--cad_models_seq', nargs='+', default=["006_mustard_bottle", "021_bleach_cleanser", "021_bleach_cleanser"])
    parser.add_argument('--risk_threshold', type=float, default=0.5, help="ADD-S风险阈值(cm)")
    parser.add_argument('--p_risk_threshold', type=float, default=0.8, help="B5概率决策阈值")
    parser.add_argument('--rollout_horizon', type=int, default=50, help="short-horizon长度，<=0表示不reset")
    parser.add_argument('--blackout_min_frames', type=int, default=10)
    args = parser.parse_args()
    main(args)
