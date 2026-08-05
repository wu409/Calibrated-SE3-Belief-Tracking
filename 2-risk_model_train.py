import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

# 1. 载入刚才导出的 CSV 数据集
csv_path = "./per_frame_harm_dataset_delta0.2.csv"
if not os.path.exists(csv_path):
    print("错误: 找不到 csv，请先运行生成 CSV 脚本！")
    exit()

df = pd.read_csv(csv_path)

# 2. 序列隔离划分 (Leave-One-Sequence-Out LOSO，防止数据泄露)
# 用 bleach0 和 mustard0 训练，用 bleach_hard 独立测试！
train_df = df[df['sequence'].str.contains("bleach0")| df['sequence'].str.contains("mustard0")]  # 
test_df = df[df['sequence'].str.contains("bleach_hard_00_03_chaitanya")]

# 如果测试集为空的防护
if len(test_df) == 0:
    print("错误: 测试集为空，请检查 CSV 文件中的序列名称！")

feature_cols = ['x1_depth_residual', 'x2_inlier_ratio','x3_innovation_mag','x4_support_ratio']# 'x3_trans_innovation', 'x3_rot_innovation',
X_train, y_train = train_df[feature_cols].values, train_df['harm_label'].values
X_test, y_test = test_df[feature_cols].values, test_df['harm_label'].values

scaler_x = MinMaxScaler()
X_train_scaled = scaler_x.fit_transform(X_train) # 训练集拟合并标准化
X_test_scaled = scaler_x.transform(X_test)      # 测试集用同样的缩放转换


# 3. 训练逻辑回归模型
print("正在训练 P(Harm) 逻辑回归分类器...")
clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X_train_scaled, y_train)

# 4. 温度缩放标定 (Temperature Scaling，保证概率不盲目自信)
train_logits = clf.decision_function(X_train_scaled)
test_logits = clf.decision_function(X_test_scaled)

def eval_loss(t):
    scaled = train_logits / t[0]
    probs = 1.0 / (1.0 + np.exp(-scaled))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return -np.mean(y_train * np.log(probs) + (1 - y_train) * np.log(1 - probs))

res = minimize(eval_loss, [1.0], bounds=[(0.01, 10.0)])
temp_factor = res.x[0]

# 预测测试集上标定后的连续概率 P(Harm) (0.0 到 1.0)
test_probs = 1.0 / (1.0 + np.exp(-(test_logits / temp_factor)))
print(f"测试集上标定后的 P(Harm) 概率范围: [{test_probs.min():.4f}, {test_probs.max():.4f}]")

# 5. ECE (预期标定误差) 计算函数
def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            ece += np.abs(np.mean(labels[in_bin]) - np.mean(probs[in_bin])) * (np.sum(in_bin) / len(probs))
    return ece

# 计算 4 大概率硬核指标
auroc = roc_auc_score(y_test, test_probs)
precision, recall, _ = precision_recall_curve(y_test, test_probs)
auprc = auc(recall, precision)
brier = brier_score_loss(y_test, test_probs)
ece = compute_ece(test_probs, y_test)

# 打印结果表
print("\n" + "="*50)
print("🌟 P(Harm) 风险模型训练与标定指标 🌟")
print("="*50)
print(f"1. AUROC (风险区分能力):       {auroc:.4f}  (越接近 1.0 越好)")
print(f"2. AUPRC (精确召回率):         {auprc:.4f}  (越接近 1.0 越好)")
print(f"3. Brier Score (均方概率误差):  {brier:.4f}  (越接近 0.0 越好)")
print(f"4. ECE (预期标定误差):         {ece:.4f}    (越接近 0.0 越好)")
print(f"5. 温度缩放系数 Temp Factor:   {temp_factor:.4f}")
print("="*50)

# 6. 绘制并保存 Harry 点名要的 Reliability Diagram (概率标定图)
plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (ECE=0)')

n_bins = 10
bin_boundaries = np.linspace(0, 1, n_bins + 1)
bin_accs, bin_confs = [], []

for i in range(n_bins):
    in_bin = (test_probs > bin_boundaries[i]) & (test_probs <= bin_boundaries[i+1])
    if np.sum(in_bin) > 0:
        bin_accs.append(np.mean(y_test[in_bin]))
        bin_confs.append(np.mean(test_probs[in_bin]))

plt.plot(bin_confs, bin_accs, 's-', color='darkorange', linewidth=2, label=f'Calibrated P(Harm) (ECE={ece:.3f})')

plt.title('Reliability Diagram for Risk Predictor $P(Harm)$', fontsize=11)
plt.xlabel('Predicted Risk Probability P(Harm)(alpha=0.2%)', fontsize=11)
plt.ylabel('Observed Harm Frequency', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('reliability_diagram.png', dpi=300, bbox_inches='tight')
print("\nHarry 点名要的标定图已成功保存为: reliability_diagram.png！")