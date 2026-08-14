import os
import glob
import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import Utils as U
from scipy.spatial.transform import Rotation as R_sci
from scipy.optimize import minimize
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
import argparse
from scipy.stats import t

K = np.array([
    [3.195820007324218750e+02,    0.0,   3.202149847676955687e+02],
    [   0.0,  4.171186828613281250e+02, 2.443486680871046701e+02],
    [   0.0,      0.0,     1.0   ]
], dtype=np.float64)

# ==================== 1. SE(3) 李群辅助函数 ====================
def se3_log_map(T):
    R_mat, t_vec = T[:3, :3], T[:3, 3]
    w_vec = R_sci.from_matrix(R_mat).as_rotvec()
    return np.concatenate([t_vec, w_vec])

def se3_exp_map(delta):
    t_vec, w_vec = delta[:3], delta[3:]
    R_mat = R_sci.from_rotvec(w_vec).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mat
    T[:3, 3] = t_vec
    return T

def compute_se3_prior(T_prev1, T_prev2):
    delta = se3_log_map(np.linalg.inv(T_prev2) @ T_prev1)
    return T_prev1 @ se3_exp_map(delta)

def calc_auc(errors_cm, max_threshold_cm=10.0):
    errors_array = np.array(errors_cm)
    if len(errors_array) == 0: return 0.0
    thresholds = np.linspace(0, max_threshold_cm, 1000)
    accs = [np.mean(errors_array <= th) for th in thresholds]
    return (np.trapz(accs, thresholds) / max_threshold_cm) * 100.0

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            ece += np.abs(np.mean(labels[in_bin]) - np.mean(probs[in_bin])) * (np.sum(in_bin) / len(probs))
    return ece

def compute_episode_level_ci(scores, n_bootstraps=2000, confidence_level=0.95):
    """按 Episode 计算 95% 置信区间 """
    scores_arr = np.array(scores)
    if len(scores_arr) < 2:
        m = np.mean(scores_arr)
        return m, m, m # 单 Episode 直接返回均值

    bootstrapped_means = []
    np.random.seed(42)
    for _ in range(n_bootstraps):
        sample = np.random.choice(scores_arr, size=len(scores_arr), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = (1.0 - confidence_level) / 2.0
    lower_bound = np.percentile(bootstrapped_means, alpha * 100)
    upper_bound = np.percentile(bootstrapped_means, (1.0 - alpha) * 100)
    mean_score = np.mean(scores_arr)

    return mean_score, lower_bound, upper_bound

def get_frame_id(path):

    filename = os.path.basename(path)
    frame_id = os.path.splitext(filename)[0]
    return int(frame_id)

def actual_recovery_action(current_depth_real, T_obs, model_pts_3d, K):  

    z_center = T_obs[2, 3]
    y_idx, x_idx = np.where((current_depth_real > z_center - 0.2) & (current_depth_real < z_center + 0.2))

    if len(y_idx) < 100:
        print("recovery的输出为T_prior")
        T = False
        return False # 依然全黑，无法恢复，继续听先验
    
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


def main(args):
    # ==================== 2. 载入数据集与训练好的风险分类器 ====================
    print("正在载入 CSV 数据集并训练分类器...")
    df = pd.read_csv(args.csv_path)
    path = args.result_dir
    test_seq = os.path.basename(path)
    train_pattern = "|".join(args.train_seqs)
    train_df = df[df['sequence'].str.contains(train_pattern)]

    matched_train_seqs = df[df['sequence'].str.contains(train_pattern)]['sequence'].unique()  
    #对匹配到的每一个训练序列，按时间轴切成前 80% (训练) 和 后 20% (校准)
    train_dfs, cal_dfs = [], []
    for seq in matched_train_seqs:
        sub_df = df[df['sequence'] == seq]
        split_idx = int(len(sub_df) * 0.8) # 80% 时间截断点
        
        train_dfs.append(sub_df.iloc[:split_idx]) # 前 80% 时间段进入训练集
        cal_dfs.append(sub_df.iloc[split_idx:])   # 后 20% 时间段进入校准集

    train_df = pd.concat(train_dfs)
    cal_df   = pd.concat(cal_dfs)

    test_df = df[df['sequence'] == test_seq].copy()
    test_df['frame_id'] = test_df['frame_id'].astype(int)
    test_df = test_df.set_index('frame_id')

    test_ALL_df = df[df['sequence'].str.contains(args.test_base_seq)]
    
    print(f"训练集包含序列关键词: {args.train_seqs} | 行数: {len(train_df)}")
    print(f"测试集 (单序列) 名称: {test_seq} | 行数: {len(test_df)}")
    print(f"测试集 (全量变体) 名称: {args.test_base_seq} | 行数: {len(test_ALL_df)}")

    if len(test_df) == 0:
        print("错误: 测试集为空，请检查 CSV 文件中的序列名称！")

    feature_cols = ['x1_depth_residual', 'x2_inlier_ratio','x3_innovation_mag','x4_support_ratio']

    X_train = train_df[feature_cols].values
    X_cal = cal_df[feature_cols].values
    X_test = test_df[feature_cols].values
    X_ALL_test = test_ALL_df[feature_cols].values

    y_obs_train = train_df['obs_risk_label'].values
    y_obs_cal = cal_df['obs_risk_label'].values
    y_obs_test = test_df['obs_risk_label'].values
    y_obs_ALL_test = test_ALL_df['obs_risk_label'].values

    y_prior_train = train_df['prior_risk_label'].values
    y_prior_cal = cal_df['prior_risk_label'].values
    y_prior_test = test_df['prior_risk_label'].values
    y_prior_ALL_test = test_ALL_df['prior_risk_label'].values

    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_cal_scaled = scaler_x.transform(X_cal)
    X_test_scaled = scaler_x.transform(X_test)
    X_test_ALL_scaled = scaler_x.transform(X_ALL_test)

    clf_obs = LogisticRegression(max_iter=1000)
    clf_obs.fit(X_train_scaled, y_obs_train)

    clf_prior = LogisticRegression(max_iter=1000)
    clf_prior.fit(X_train_scaled,y_prior_train)

    # 4. 温度缩放标定 (Temperature Scaling，保证概率不盲目自信)
    cal_logits_obs = clf_obs.decision_function(X_cal_scaled)
    test_logits_obs = clf_obs.decision_function(X_test_scaled)
    test_ALL_logits_obs = clf_obs.decision_function(X_test_ALL_scaled)   #用于绘制ECE、auroc、auprc、brier

    cal_logits_prior = clf_prior.decision_function(X_cal_scaled)
    test_logits_prior = clf_prior.decision_function(X_test_scaled)
    test_ALL_logits_prior = clf_prior.decision_function(X_test_ALL_scaled)


    def eval_loss_obs(t):
        scaled = cal_logits_obs / t[0]
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(y_obs_cal * np.log(probs) + (1 - y_obs_cal) * np.log(1 - probs))
    
    def eval_loss_prior(t):
            scaled = cal_logits_prior / t[0]
            probs = 1.0 / (1.0 + np.exp(-scaled))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(y_prior_cal * np.log(probs) + (1 - y_prior_cal) * np.log(1 - probs))

    res_obs = minimize(eval_loss_obs, [1.0], bounds=[(0.01, 10.0)])
    temp_factor_obs = res_obs.x[0]

    res_prior = minimize(eval_loss_prior, [1.0], bounds=[(0.01, 10.0)])
    temp_factor_prior = res_prior.x[0]

    p_obs_bad = 1.0 / (1.0 + np.exp(-(test_logits_obs / temp_factor_obs)))
    p_obs_bad_ALL = 1.0 / (1.0 + np.exp(-(test_ALL_logits_obs / temp_factor_obs)))

    p_prior_bad = 1.0 / (1.0 + np.exp(-(test_logits_prior / temp_factor_prior)))
    p_prior_bad_ALL = 1.0 / (1.0 + np.exp(-(test_ALL_logits_prior / temp_factor_prior)))

    p_obs_bad_dict = dict(zip(test_df.index.tolist(),p_obs_bad.tolist()))
    p_prior_bad_dict = dict(zip(test_df.index.tolist(),p_prior_bad.tolist()))
    support_dict = test_df['x4_support_ratio'].to_dict()

    for fid in list(test_df.index[:10]):
        print(
            f"frame_id={fid:4d} | "
            f"P(obs_bad)={p_obs_bad_dict[fid]:.4f} | "
            f"P(prior_bad)={p_prior_bad_dict[fid]:.4f} | "
            f"support={support_dict[fid]:.4f}"
        )
    
    # ==================== 3. 计算 4 大概率与标定指标 ====================
    auroc_obs = roc_auc_score(y_obs_ALL_test, p_obs_bad_ALL)
    precision_obs, recall_obs, _ = precision_recall_curve(y_obs_ALL_test, p_obs_bad_ALL)
    auprc_obs = auc(recall_obs, precision_obs)
    brier_obs = brier_score_loss(y_obs_ALL_test, p_obs_bad_ALL)
    ece_obs = compute_ece(p_obs_bad_ALL,y_obs_ALL_test)

    auroc_prior = roc_auc_score(y_prior_ALL_test, p_prior_bad_ALL)
    precision_prior, recall_prior, _ = precision_recall_curve(y_prior_ALL_test, p_prior_bad_ALL)
    auprc_prior = auc(recall_prior, precision_prior)
    brier_prior = brier_score_loss(y_prior_ALL_test, p_prior_bad_ALL)
    ece_prior = compute_ece(p_prior_bad_ALL, y_prior_ALL_test)


    # ==================== 5. 运行 6 个 Baselines====================
    print("正在测试序列上运行 6 个 Baselines PK...")
    point_path = args.point_path
    with open(point_path, 'r') as f:
        model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
    open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))


    result_dir=args.result_dir
    gt_dir=args.gt_dir
    last_name = os.path.basename(args.result_dir)

    pred_files=glob.glob(os.path.join(result_dir,"*.txt"))
    gt_files=glob.glob(os.path.join(gt_dir,"*.txt"))

    pred_dict = {get_frame_id(f): f for f in pred_files}
    gt_dict = {get_frame_id(f): f for f in gt_files}

    depth_dict = build_safe_depth_dict(f"{args.data_dir}/{last_name}/depth", gt_dict)
    matched_frames = sorted(set(pred_dict.keys()) & set(gt_dict.keys()) & set(depth_dict.keys()))
    print(f"成功安全对齐帧数: {len(matched_frames)} 帧！")

    # 显式求交集
    matched_frames = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))
    missing_pred = sorted(set(gt_dict.keys()) - set(pred_dict.keys()))
    missing_gt = sorted(set(pred_dict.keys()) - set(gt_dict.keys()))

    print("="*50)
    print(f"GT frames        : {len(gt_dict)}")
    print(f"Prediction frames: {len(pred_dict)}")
    print(f"Matched frames   : {len(matched_frames)}")
    print(f"Missing prediction frames:")
    print(missing_pred[:20])
    print(f"Missing GT frames:")
    print(missing_gt[:20])
    print("="*50)


    T_candidates=[]
    T_gts=[]
    frame_ids=[]


    for frame_id in matched_frames:
        pred_path = pred_dict[frame_id]
        gt_path   = gt_dict[frame_id]
        T_pred = np.loadtxt(pred_path).reshape(4,4)
        T_gt   = np.loadtxt(gt_path).reshape(4,4)
        T_candidates.append(T_pred)
        T_gts.append(T_gt)
        frame_ids.append(frame_id)
  

    T_candidates=np.array(T_candidates)
    T_gts=np.array(T_gts)
    frame_ids=np.array(frame_ids)


    # 定义 6 个 Baselines 最终的逐帧姿态误差结果 (cm)
    b1_errs,  b2_errs,  b3_errs,  b4_errs,  b5_errs,  b6_errs  = [], [], [], [], [], []
    b5_modes = [] # 记录三模式历史

    consecutive_blackout = 0
    false_recovery_triggers = 0
    recovery_latencies1,recovery_latencies2,recovery_latencies3,recovery_latencies4,recovery_latencies5,recovery_latencies6 = [],[],[],[],[],[] # 记录恢复延迟
    T_history2,T_history3,T_history4,T_history5,T_history6 =  [], [], [], [], []
    blackout_start = 0
    blackout_end = 1e10


    for i, frame_id in enumerate(matched_frames):

        if i < 2: 
            T_prior_current2,T_prior_current3,T_prior_current4,T_prior_current5,T_prior_current6 = T_candidates[i],T_candidates[i],T_candidates[i],T_candidates[i],T_candidates[i]  # 初始化阶段
        else:
            T_prior_current2 = compute_se3_prior(T_history2[-1],T_history2[-2])
            T_prior_current3 = compute_se3_prior(T_history3[-1],T_history3[-2])
            T_prior_current4 = compute_se3_prior(T_history4[-1],T_history4[-2])
            T_prior_current5 = compute_se3_prior(T_history5[-1],T_history5[-2])
            T_prior_current6 = compute_se3_prior(T_history6[-1],T_history6[-2])
   
        p_obs_bad = p_obs_bad_dict[frame_id]
        p_prior_bad = p_prior_bad_dict[frame_id]
        support = support_dict[frame_id]
        # B1: Obs-Only 
        #b1_errs.append(e_obs)
        b1_errs.append(U.adi(T_candidates[i],T_gts[i],open3d_model)* 100)
        
        # B2: Fixed-alpha Smoothing (0.5)
        delta=se3_log_map(np.linalg.inv(T_prior_current2)@T_candidates[i])
        T_final2=T_prior_current2 @ se3_exp_map(args.alpha*delta)
        b2_errs.append(U.adi(T_final2,T_gts[i],open3d_model)* 100)
        T_history2.append(T_final2)

        # B3: Hard Depth Threshold (深度缺失则听惯性)
        if support<0.4:
            T_final3 = T_candidates[i]
        else:
            T_final3 = T_prior_current3
        T_history3.append(T_final3)
        b3_errs.append(U.adi(T_final3,T_gts[i],open3d_model)* 100)

        # B4: Robust Huber Weighting
        innovation=se3_log_map(np.linalg.inv(T_prior_current4)@T_candidates[i])
        r=np.linalg.norm(innovation)
        delta=0.1
        if r<=delta:
            alpha=1.0
        else:
            alpha=delta/r
        T_huber=T_prior_current4@se3_exp_map(alpha*innovation)
        T_history4.append(T_huber)
        b4_errs.append(U.adi(T_huber,T_gts[i],open3d_model)* 100)

        #  B5: Proposed Three-Mode Policy
        if support == 1:
            consecutive_blackout += 1
            if consecutive_blackout==1:
                blackout_start = i
        else:
            if consecutive_blackout >= 10:
                exited_blackout = True # 标记黑屏刚刚结束，准备触发recovery！
                blackout_end = i
            consecutive_blackout = 0

        if support == 1:
            #print("黑屏帧数",i)
            # 只要当前帧是黑屏，只能听惯性
            current_mode = "MODE_3_BLACKOUT_WAITING"
            T_final = T_prior_current5
            #print(p_help)

        elif 'exited_blackout' in locals() and exited_blackout:
            print("恢复帧数",frame_id)
            depth_raw = cv2.imread(depth_dict[frame_id],cv2.IMREAD_UNCHANGED)
            depth_real = depth_raw.astype(np.float32) / 1000.0
            T_final, T = actual_recovery_action(depth_real,T_candidates[i], model_pts, K)
            if not T:
                T_final=T_prior_current5
            current_mode = "MODE_3_RECOVERY_EXECUTE"
            exited_blackout=False

        # =========================
        # 正常模式
        # =========================
        elif p_obs_bad <= 0.7:
            # if 55<i:
            #  print(i)
            current_mode = "MODE_1_ACCEPT"
            T_final = T_candidates[i]
        else:
            current_mode = "MODE_2_PRIOR"
            T_final = T_prior_current5 

        T_history5.append(T_final)
        b5_errs.append(U.adi(T_final, T_gts[i], open3d_model) * 100)
        b5_modes.append(current_mode)

        if (i < blackout_start or i>blackout_end) and current_mode == "MODE_3_BLACKOUT_WAITING" :
                print(blackout_start)
                print(blackout_end)
                print(i)
                false_recovery_triggers += 1


        # B6: Oracle Decision Policy
        err_obs = U.adi(T_candidates[i], T_gts[i],open3d_model)
        err_prior = U.adi(T_prior_current6,T_gts[i],open3d_model)
        if err_obs < err_prior:
            T_final_oracle = T_candidates[i]
        else:
            T_final_oracle = T_prior_current6
        T_history6.append(T_final_oracle)
        b6_errs.append(U.adi(T_final_oracle,T_gts[i],open3d_model)*100)

        # T_final2=T_prior_current2
        # T_history2.append(T_final2)
        #b7.append(U.adi(T_final2,T_gts[i],open3d_model)*100)
       
    print(Counter(b5_modes))
    print("black_start:",blackout_start)
    print("black_end:",blackout_end)
    # print(b5_errs[42:200])

    # ==================== 动作 A: 测量 【恢复延迟 Recovery Latency】 ====================
    # 找到黑屏结束点 (第 160 帧)，测量复活需要几帧
    a=0.5
    if "black" in args.result_dir:
        print("computing recovery_latencies")
        for i in range(blackout_end, len(b5_errs)):
            if b1_errs[i] < a: 
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies1.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b2_errs[i] < a: 
                
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies2.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b3_errs[i] < a: 

                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies3.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b4_errs[i] < a: 
                
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies4.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b5_errs[i] < a: 
                
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies5.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b6_errs[i] < a: 
                
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies6.append(latency)
                break


    if len(recovery_latencies5) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency5  = f"{recovery_latencies5[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency5  = "N/A (Failed)"

    if len(recovery_latencies1) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency1  = f"{recovery_latencies1[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency1  = "N/A (Failed)"

    if len(recovery_latencies2) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency2  = f"{recovery_latencies2[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency2  = "N/A (Failed)"

    if len(recovery_latencies3) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency3  = f"{recovery_latencies3[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency3  = "N/A (Failed)"

    if len(recovery_latencies4) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency4  = f"{recovery_latencies4[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency4  = "N/A (Failed)"

    if len(recovery_latencies6) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency6  = f"{recovery_latencies6[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency6  = "N/A (Failed)"

    # ==================== 动作 C: 保存逐帧 CSV 日志与黑屏复活轨迹图 ====================
    # ==================== 6. 保存图像与 CSV 日志 ====================
    # 保存逐帧 CSV 日志
    df_log = pd.DataFrame({
        "frame_id": matched_frames,
        "p_obs_bad": [p_obs_bad_dict[f] for f in matched_frames],
        "p_prior_bad": [p_prior_bad_dict[f] for f in matched_frames],
        "selected_mode": b5_modes,
        "error_b1_obs_cm": b1_errs,
        "error_b2_obs_cm": b2_errs,
        "error_b3_obs_cm": b3_errs,
        "error_b4_obs_cm": b4_errs,
        "error_b5_ours_cm": b5_errs,
        "error_b6_obs_cm": b6_errs,

    })
    df_log.to_csv(f"./checkpoint2_per_frame_{last_name}_log_threshold{args.risk_threshold}.csv", index=False)

    # 保存 Reliability Diagram
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (ECE=0)')
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        in_bin = (p_obs_bad_ALL > bin_boundaries[i]) & (p_obs_bad_ALL <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(y_obs_ALL_test[in_bin]))
            bin_confs.append(np.mean(p_obs_bad_ALL[in_bin]))
    plt.plot(bin_confs, bin_accs, 's-', color='darkorange', linewidth=2, label=f'Ours (ECE={ece_obs:.3f})')
    plt.title(r'Reliability Diagram for Observation Risk', fontsize=11)
    plt.xlabel(r'Predicted $P(\mathrm{Observation\ Bad})$', fontsize=11)
    plt.ylabel('Empirical Observed Help Frequency', fontsize=11)
    plt.legend(); plt.grid(True, linestyle='--')
    plt.savefig(f'reliability_diagram_observation_risk_threshold{args.risk_threshold}.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (ECE=0)')
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        in_bin = (p_prior_bad_ALL > bin_boundaries[i]) & (p_prior_bad_ALL <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(y_prior_ALL_test[in_bin]))
            bin_confs.append(np.mean(p_prior_bad_ALL[in_bin]))
    plt.plot(bin_confs, bin_accs, 's-', color='darkorange', linewidth=2, label=f'Ours (ECE={ece_prior:.3f})')
    plt.title(r'Reliability Diagram for Prior Risk', fontsize=11)
    plt.xlabel(r'Predicted $P(\mathrm{Prior\ Bad})$', fontsize=11)
    plt.ylabel('Empirical Observed Help Frequency', fontsize=11)
    plt.legend(); plt.grid(True, linestyle='--')
    plt.savefig(f'reliability_diagram_prior_risk_threshold{args.risk_threshold}.png', dpi=300, bbox_inches='tight')

    if "black" in args.result_dir: 
        # 保存 Trajectory Trace Plot
        plt.figure(figsize=(10, 5))
        plt.plot(b1_errs[:], 'r-', label='B1: Obs-Only (Diverges on Blackout)', alpha=0.7)
        plt.plot(b5_errs[:], 'g--', label='B5: Ours (Three-Mode Policy)', linewidth=2)
        plt.axvspan(blackout_start, blackout_end, color='gray', alpha=0.3, label='10-Frame Complete Blackout')
        plt.title('Temporal Trajectory Trace: Pose Error & Recovery under Blackout', fontsize=12)
        plt.xlabel('Frame Number', fontsize=11)
        plt.ylabel('ADD-S Pose Error (cm)', fontsize=11)
        plt.legend(); plt.grid(True, linestyle='--')
        plt.savefig(f'trajectory_recovery_plot_threshold{args.risk_threshold}.png', dpi=300, bbox_inches='tight')

    print("\n所有的 11 项交付物已全部生成完毕！图片与 CSV 已成功保存！")


    adds_scores = [calc_auc(b1_errs), calc_auc(b2_errs), calc_auc(b3_errs), calc_auc(b4_errs), calc_auc(b5_errs), calc_auc(b6_errs)]
    fail_rates = [np.mean(np.array(b1_errs) > 2.0)*100, np.mean(np.array(b2_errs) > 2.0)*100,np.mean(np.array(b3_errs) > 2.0)*100, np.mean(np.array(b4_errs) > 2.0)*100,np.mean(np.array(b5_errs) > 2.0)*100, np.mean(np.array(b6_errs) > 2.0)*100]
    latency_scores = [f"{avg_recovery_latency1} frames", f"{avg_recovery_latency2} frames", f"{avg_recovery_latency3} frames", f"{avg_recovery_latency4} frames", f"{avg_recovery_latency5} frames", f"{avg_recovery_latency6} frames"]
    false_triggers = ["N/A", "N/A", "N/A", "N/A", f"{false_recovery_triggers} times", "0 times"]
    prob_metrics_obs = {'auroc_obs': auroc_obs,'auprc_obs': auprc_obs,'brier_obs': brier_obs,'ece_obs': ece_obs,'temp_factor_obs': temp_factor_obs} 
    prob_metrics_prior = {'auroc_prior': auroc_prior,'auprc_prior': auprc_prior,'brier_prior': brier_prior,'ece_prior': ece_prior,'temp_factor_prior': temp_factor_prior} 
    

    return adds_scores, fail_rates, latency_scores, false_triggers, prob_metrics_obs, prob_metrics_prior
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="./per_frame_label_threshold0.5.csv", help="csv数据集路径")
    parser.add_argument('--result_dir', nargs='+',type=str, default=["./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10",
                                                                     "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_2",
                                                                     "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_3",
                                                                     "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_4",
                                                                     "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_5",], 
                                                                     help="要测试的所有序列路径")
    parser.add_argument('--gt_dir', type=str, default="./datasets/YCBInEOAT/bleach_hard_00_03_chaitanya/annotated_poses", help="GT_Pose Path")
    parser.add_argument('--point_path', type=str, default="./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz", help="point_path")
    parser.add_argument('--train_seqs', nargs='+', default=["bleach0", "mustard0"], help="训练集包含的序列关键字列表")
    parser.add_argument('--test_base_seq', type=str, default="bleach_hard_00_03_chaitanya", help="测试集物体的基础名称")
    parser.add_argument('--data_dir', type=str, default="./datasets/YCBInEOAT_Corrupted", help="受损数据集基础路径")
    parser.add_argument('--p_help_threshold', type=float, default=0.50, help="p_help_threshold")
    parser.add_argument('--alpha', type=float, default=0.5, help="B2-alpha")
    parser.add_argument('--risk_threshold', type=float, default=0.5, help="risk_threshold")
    args = parser.parse_args()
    baseline_names = [
        "B1: Obs-Only se(3)-TrackNet",
        "B2: Fixed-Alpha (0.5) Interpolation",
        "B3: Hard Depth Threshold",
        "B4: Robust Huber Weighting",
        "B5: Proposed Three-Mode Policy (Ours)",
        "B6: Oracle Decision Policy (Upper Bound)"
    ]

    ADDS_dict = {}
    FAIL_dict = {}
    LATENCY_dict = {}
    TRIGGER_dict = {}
    last_prob_metrics_obs = None # 用于保存概率标定指标
    last_prob_metrics_prior = None

    result_dirs = args.result_dir 
    
    for single_dir in result_dirs:
        args.result_dir = single_dir 
        basename = os.path.basename(single_dir)
        
        print(f"\n正在评估 Episode: {basename} ...")
        
        # 接收 main(args) 返回的 5 个元组/列表！
        adds_s, fails, lats, fts, prob_metrics_obs, prob_metrics_prior = main(args) 
        
        ADDS_dict[basename] = adds_s
        FAIL_dict[basename] = fails
        LATENCY_dict[basename] = lats
        TRIGGER_dict[basename] = fts
        last_prob_metrics_obs = prob_metrics_obs # 记录概率指标
        last_prob_metrics_prior = prob_metrics_prior 

    print("\n" + "="*100)
    print("="*100)

    summary_rows = []

    for b_idx, b_name in enumerate(baseline_names):
        scores_for_this_baseline = [ADDS_dict[ep][b_idx] for ep in ADDS_dict.keys()]
        mean_score, ci_low, ci_high = compute_episode_level_ci(scores_for_this_baseline)
        
        row_data = {"Baseline / Method": b_name}
        
        # 1. ADD-S AUC (%)
        for ep in ADDS_dict.keys():
            row_data[f"ADD-S ({ep})"] = f"{ADDS_dict[ep][b_idx]:.3f}%"
        row_data["Mean ADD-S (%)"] = f"{mean_score:.3f}%"
        row_data["95% CI"] = f"[{ci_low:.3f}%, {ci_high:.3f}%]"

        # 2. 失效率 Failure Rate (%)
        for ep in FAIL_dict.keys():
            row_data[f"Failure Rate ({ep})"] = f"{FAIL_dict[ep][b_idx]:.3f}%"

        # 3. 恢复延迟 Recovery Latency
        for ep in LATENCY_dict.keys():
            row_data[f"Latency ({ep})"] = LATENCY_dict[ep][b_idx]

        # 4. 假触发数 False Triggers
        for ep in TRIGGER_dict.keys():
            row_data[f"False Triggers ({ep})"] = TRIGGER_dict[ep][b_idx]

        summary_rows.append(row_data)

    df_summary = pd.DataFrame(summary_rows)

    # ====================  5 大概率标定指标！ ====================
    if last_prob_metrics_obs is not None and last_prob_metrics_prior is not None:
        prob_df = pd.DataFrame([
            {"Metric": "1. AUROC_obs (Risk Discrimination)",       "Value": f"{last_prob_metrics_obs['auroc_obs']:.4f}"},
            {"Metric": "2. AUPRC_obs (Precision-Recall AUC)",     "Value": f"{last_prob_metrics_obs['auprc_obs']:.4f}"},
            {"Metric": "3. Brier Score_obs (Probability MSE)",    "Value": f"{last_prob_metrics_obs['brier_obs']:.4f}"},
            {"Metric": "4. ECE_obs (Expected Calibration Error)", "Value": f"{last_prob_metrics_obs['ece_obs']:.4f}"},
            {"Metric": "5. Temp_Factor_obs (Temperature Scalar)", "Value": f"{last_prob_metrics_obs['temp_factor_obs']:.4f}"},
            {"Metric": "1. AUROC_prior (Risk Discrimination)",       "Value": f"{last_prob_metrics_prior['auroc_prior']:.4f}"},
            {"Metric": "2. AUPRC_prior (Precision-Recall AUC)",     "Value": f"{last_prob_metrics_prior['auprc_prior']:.4f}"},
            {"Metric": "3. Brier Score_prior (Probability MSE)",    "Value": f"{last_prob_metrics_prior['brier_prior']:.4f}"},
            {"Metric": "4. ECE_prior (Expected Calibration Error)", "Value": f"{last_prob_metrics_prior['ece_prior']:.4f}"},
            {"Metric": "5. Temp_Factor_prior (Temperature Scalar)", "Value": f"{last_prob_metrics_prior['temp_factor_prior']:.4f}"}
        ])

        prob_csv_path = f"./checkpoint2_probability_calibration_metrics_threshold{args.risk_threshold}.csv"
        prob_df.to_csv(prob_csv_path, index=False)
        print(f"\n✅ 5 大概率标定指标已成功导出为 CSV: {prob_csv_path}")

    # 5. 导出为全指标大 CSV 文件
    basename2 = os.path.basename(result_dirs[0])
    csv_out_path = f"./checkpoint2_full_metrics_episode_summary_{basename2}_threshold{args.risk_threshold}.csv"
    df_summary.to_csv(csv_out_path, index=False)
    print(f"\n全指标 Episode 级汇总表格已保存为: {csv_out_path}")