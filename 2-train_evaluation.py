import os
import glob
import numpy as np
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


def main(args):
    # ==================== 2. 载入数据集与训练好的风险分类器 ====================
    print("正在载入 CSV 数据集并训练分类器...")
    df = pd.read_csv(args.csv_path)
    path = args.result_dir
    test_seq = os.path.basename(path)
    train_pattern = "|".join(args.train_seqs)
    train_df = df[df['sequence'].str.contains(train_pattern)]

    matched_train_seqs = df[df['sequence'].str.contains(train_pattern)]['sequence'].unique()  
    #3. 对匹配到的每一个训练序列，按时间轴切成前 80% (训练) 和 后 20% (校准)
    train_dfs, cal_dfs = [], []
    for seq in matched_train_seqs:
        sub_df = df[df['sequence'] == seq]
        split_idx = int(len(sub_df) * 0.8) # 80% 时间截断点
        
        train_dfs.append(sub_df.iloc[:split_idx]) # 前 80% 时间段进入训练集
        cal_dfs.append(sub_df.iloc[split_idx:])   # 后 20% 时间段进入校准集

    train_df = pd.concat(train_dfs)
    cal_df   = pd.concat(cal_dfs)

    test_df = df[df['sequence'].str.contains(test_seq)]
    test_ALL_df = df[df['sequence'].str.contains(args.test_base_seq)]
    
    print(f"训练集包含序列关键词: {args.train_seqs} | 行数: {len(train_df)}")
    print(f"测试集 (单序列) 名称: {test_seq} | 行数: {len(test_df)}")
    print(f"测试集 (全量变体) 名称: {args.test_base_seq} | 行数: {len(test_ALL_df)}")

    if len(test_df) == 0:
        print("错误: 测试集为空，请检查 CSV 文件中的序列名称！")

    feature_cols = ['x1_depth_residual', 'x2_inlier_ratio','x3_innovation_mag','x4_support_ratio']

    X_train, y_train = train_df[feature_cols].values, train_df['help_label'].values
    X_cal, y_cal     = cal_df[feature_cols].values,   cal_df['help_label'].values
    X_test, y_test = test_df[feature_cols].values, test_df['help_label'].values
    X_ALL_test, y_ALL_test = test_ALL_df[feature_cols].values, test_ALL_df['help_label'].values

    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_cal_scaled = scaler_x.transform(X_cal)
    X_test_scaled = scaler_x.transform(X_test)
    X_test_ALL_scaled = scaler_x.transform(X_ALL_test)

    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)

    # 4. 温度缩放标定 (Temperature Scaling，保证概率不盲目自信)
    cal_logits = clf.decision_function(X_cal_scaled)
    test_logits = clf.decision_function(X_test_scaled)              #用于测试P(help)
    test_ALL_logits = clf.decision_function(X_test_ALL_scaled)   #用于绘制ECE、auroc、auprc、brier

    def eval_loss(t):
        scaled = cal_logits / t[0]
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(y_cal * np.log(probs) + (1 - y_cal) * np.log(1 - probs))

    res = minimize(eval_loss, [1.0], bounds=[(0.01, 10.0)])
    temp_factor = res.x[0]

    # 预测测试集上标定后的连续概率 P(Help) 
    test_probs = 1.0 / (1.0 + np.exp(-(test_logits / temp_factor)))
    test_probs_ALL = 1.0 / (1.0 + np.exp(-(test_ALL_logits / temp_factor)))



    # ==================== 3. 计算 4 大概率与标定指标 ====================
    auroc = roc_auc_score(y_ALL_test, test_probs_ALL)
    precision, recall, _ = precision_recall_curve(y_ALL_test, test_probs_ALL)
    auprc = auc(recall, precision)
    brier = brier_score_loss(y_ALL_test, test_probs_ALL)
    ece = compute_ece(test_probs_ALL, y_ALL_test)

    # ==================== 5. 运行 6 个 Baselines====================
    print("正在测试序列上运行 6 个 Baselines PK...")
    mesh_path = args.mesh_path
    with open(mesh_path, 'r') as f:
        model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
    open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))

    # 物体直径 cm
    result_dir=args.result_dir
    gt_dir=args.gt_dir

    pred_files=sorted(glob.glob(os.path.join(result_dir,"*.txt")))
    gt_files=sorted(glob.glob(os.path.join(gt_dir,"*.txt")))

    assert len(pred_files)==len(gt_files)

    T_candidates=[]
    T_gts=[]

    for pf,gf in zip(pred_files,gt_files):
        T_candidates.append(np.loadtxt(pf).reshape(4,4))
        T_gts.append(np.loadtxt(gf).reshape(4,4))


    T_priors=[]
    for i in range(len(T_candidates)):
        if i<2:
            T_priors.append(T_candidates[i])
        else:
            T_priors.append(compute_se3_prior(T_candidates[i-1],T_candidates[i-2]))


    e_obs_cm = test_df['E_update_cm'].values
    e_prior_cm = test_df['E_prior_cm'].values
    x4_support = test_df['x4_support_ratio'].values
    n_frames = len(test_df)


    # 定义 6 个 Baselines 最终的逐帧姿态误差结果 (cm)
    b1_poses, b2_poses, b3_poses, b4_poses, b5_poses, b6_poses = [], [], [], [], [], []
    b1_errs,  b2_errs,  b3_errs,  b4_errs,  b5_errs,  b6_errs,b7  = [], [], [], [], [], [],[]
    b5_modes = [] # 记录三模式历史

    consecutive_blackout = 0
    in_recovery = False
    false_recovery_triggers = 0
    recovery_latencies1,recovery_latencies2,recovery_latencies3,recovery_latencies4,recovery_latencies5,recovery_latencies6 = [],[],[],[],[],[] # 记录恢复延迟
    T_history=[]
    T_history2=[]
    blackout_start = 0
    blackout_end = 0
    for i in range(n_frames):

        if i < 2:
            T_prior_current = T_candidates[i]  # 初始化阶段
            #T_prior_current2= T_candidates[i]
        else:
            T_prior_current = compute_se3_prior(T_history[i-1],T_history[i-2])
            #T_prior_current2 = compute_se3_prior(T_history2[i-1],T_history2[i-2])

        e_obs = e_obs_cm[i]
        e_prior = e_prior_cm[i]
        p_help = test_probs[i]
        support = x4_support[i]
        T_candidate = T_candidates[i]
        # B1: Obs-Only 
        #b1_errs.append(e_obs)
        b1_errs.append(U.adi(T_candidates[i],T_gts[i],open3d_model)* 100)

        # B2: Fixed-alpha Smoothing (0.5)
        delta=se3_log_map(np.linalg.inv(T_priors[i])@T_candidates[i])
        T_alpha=T_priors[i]@se3_exp_map(args.alpha*delta)
        b2_errs.append(U.adi(T_alpha,T_gts[i],open3d_model)* 100)

        # B3: Hard Depth Threshold (深度缺失则听惯性)
        b3_errs.append(e_obs if support < 0.2 else e_prior)

        # B4: Robust Huber Weighting
        innovation=se3_log_map(np.linalg.inv(T_priors[i])@T_candidates[i])
        r=np.linalg.norm(innovation)
        delta=0.1
        if r<=delta:
            alpha=1.0
        else:
            alpha=delta/r
        T_huber=T_priors[i]@se3_exp_map(alpha*innovation)
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
            T_final = T_prior_current
            print(p_help)

        elif 'exited_blackout' in locals() and exited_blackout:
            print("恢复帧数",i)
            #开灯的第一帧，用 T_obs 强行重置姿态复活！
            current_mode = "MODE_3_RECOVERY_EXECUTE"
            T_final = T_candidate            # 强制用第 160 帧新鲜的 T_obs 重置姿态！
            exited_blackout = False # 复活完成，标记重置！

        # =========================
        # 正常模式
        # =========================
        elif p_help > args.p_help_threshold:
            # if 55<i:
            #  print(i)
            current_mode = "MODE_1_ACCEPT"
            T_final = T_candidate

        else:
            current_mode = "MODE_2_PRIOR"
            T_final = T_prior_current

        # if e_obs + 0.1/27.88 > e_prior:
        #     T_final=T_candidate
        #     current_mode = "MODE_1_Accept"
        # else:
        #     current_mode = "MODE_2_PRIOR"
        #     T_final=T_prior_current    

        T_history.append(T_final)
        err_b5 = U.adi(T_final, T_gts[i], open3d_model) * 100
        b5_errs.append(err_b5)
        b5_modes.append(current_mode)

        if i< blackout_start or i>blackout_end and b5_modes == "MODE_3_BLACKOUT_WAITING" :
                false_recovery_triggers += 1

        # B6: Oracle Decision Policy
        err_obs = U.adi(T_candidate, T_gts[i],open3d_model)
        err_prior = U.adi(T_prior_current,T_gts[i],open3d_model)
        if err_obs < err_prior:
            T_final_oracle = T_candidate
        else:
            T_final_oracle = T_prior_current
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

    # ==================== 动作 B: 打印 Table 1 汇总表 ====================
    print("\n" + "="*85)
    print("TABLE 1: ALL METRICS INCLUDED")
    print("="*85)

    results_table = pd.DataFrame({
        "Baseline / Method": [
            "B1: Obs-Only se(3)-TrackNet",
            "B2: Fixed-Alpha (0.5) Interpolation",
            "B3: Hard Depth Threshold",
            "B4: Robust Huber Weighting",
            "B5: Proposed Three-Mode Policy (Ours)",
            "B6: Oracle Decision Policy (Upper Bound)"
        ],
        "ADD-S AUC (%)": [
            calc_auc(b1_errs), calc_auc(b2_errs), calc_auc(b3_errs),
            calc_auc(b4_errs), calc_auc(b5_errs), calc_auc(b6_errs)
        ],
        "Failure Rate (>2cm %)": [
            np.mean(np.array(b1_errs) > 2.0)*100, np.mean(np.array(b2_errs) > 2.0)*100,
            np.mean(np.array(b3_errs) > 2.0)*100, np.mean(np.array(b4_errs) > 2.0)*100,
            np.mean(np.array(b5_errs) > 2.0)*100, np.mean(np.array(b6_errs) > 2.0)*100
        ],
        "Recovery Latency (Frames)": [
        f"{avg_recovery_latency1} frames", f"{avg_recovery_latency2} frames", f"{avg_recovery_latency3} frames", f"{avg_recovery_latency4} frames", f"{avg_recovery_latency5} frames", f"{avg_recovery_latency6} frames"
        ],
        "False Triggers": [
            "N/A", "N/A", "N/A", "N/A", f"{false_recovery_triggers} times", "0 times"
        ]
    })

    print(results_table.to_string(index=False))
    print("="*85)


    print("\n" + "="*50)
    print("概率标定与风险预测指标")
    print("="*50)
    print(f"1. AUROC (风险区分能力):       {auroc:.4f}")
    print(f"2. AUPRC (精确召回率):         {auprc:.4f}")
    print(f"3. Brier Score (均方概率误差):  {brier:.4f}")
    print(f"4. ECE (预期标定误差):         {ece:.4f}")
    print(f"5. Temp_Factor:{temp_factor:.4F}")
    print("="*50)


    # ==================== 动作 C: 保存逐帧 CSV 日志与黑屏复活轨迹图 ====================
    # ==================== 6. 保存图像与 CSV 日志 ====================
    # 保存逐帧 CSV 日志
    df_log = pd.DataFrame({
        "frame_idx": range(len(b5_errs)),
        "p_help": test_probs[:len(b5_errs)],
        "selected_mode": b5_modes,
        "error_b1_obs_cm": b1_errs,
        "error_b5_ours_cm": b5_errs
    })
    df_log.to_csv("./checkpoint2_per_frame_rollout_log.csv", index=False)

    # 保存 Reliability Diagram
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (ECE=0)')
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        in_bin = (test_probs_ALL > bin_boundaries[i]) & (test_probs_ALL <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(y_ALL_test[in_bin]))
            bin_confs.append(np.mean(test_probs_ALL[in_bin]))
    plt.plot(bin_confs, bin_accs, 's-', color='darkorange', linewidth=2, label=f'Ours (ECE={ece:.3f})')
    plt.title(r'Reliability Diagram for Risk Predictor $P(\mathrm{Help})$', fontsize=11)
    plt.xlabel(r'Predicted Risk Probability $P(\mathrm{Help})$', fontsize=11)
    plt.ylabel('Empirical Observed Help Frequency', fontsize=11)
    plt.legend(); plt.grid(True, linestyle='--')
    plt.savefig('reliability_diagram_checkpoint2.png', dpi=300, bbox_inches='tight')

    # 保存 Trajectory Trace Plot
    plt.figure(figsize=(10, 5))
    plt.plot(b1_errs[:], 'r-', label='B1: Obs-Only (Diverges on Blackout)', alpha=0.7)
    plt.plot(b5_errs[:], 'g--', label='B5: Ours (Three-Mode Policy)', linewidth=2)
    plt.axvspan(blackout_start, blackout_end, color='gray', alpha=0.3, label='10-Frame Complete Blackout')
    plt.title('Temporal Trajectory Trace: Pose Error & Recovery under Blackout', fontsize=12)
    plt.xlabel('Frame Number', fontsize=11)
    plt.ylabel('ADD-S Pose Error (cm)', fontsize=11)
    plt.legend(); plt.grid(True, linestyle='--')
    plt.savefig('trajectory_recovery_plot.png', dpi=300, bbox_inches='tight')

    print("\n所有的 11 项交付物已全部生成完毕！图片与 CSV 已成功保存！")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="./per_frame_help_dataset_delta0.csv", help="csv数据集路径")
    parser.add_argument('--result_dir', type=str, default="./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10", help="要测试的单个序列路径")
    parser.add_argument('--gt_dir', type=str, default="./datasets/YCBInEOAT/bleach_hard_00_03_chaitanya/annotated_poses", help="GT_Pose Path")
    parser.add_argument('--mesh_path', type=str, default="./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz", help="mesh_path")
    parser.add_argument('--train_seqs', nargs='+', default=["bleach0", "mustard0"], help="训练集包含的序列关键字列表")
    parser.add_argument('--test_base_seq', type=str, default="bleach_hard_00_03_chaitanya", help="测试集物体的基础名称")
    parser.add_argument('--p_help_threshold', type=float, default=0.60, help="p_help_threshold")
    parser.add_argument('--alpha', type=float, default=0.5, help="B2-alpha")
    args = parser.parse_args()
    main(args)
