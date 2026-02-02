# -*- coding: utf-8 -*-
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold, train_test_split

# 导入自定义模型、损失函数和工具函数
from model import DualGraphRNAModel
from predict import tensor_to_degrees

# ==========================================
# 1. 全局配置与超参数
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 10.0
BATCH_SIZE = 1 # 建议调大一点，1 太慢且不稳定
NUM_EPOCHS = 100
PATIENCE = 60
PRED_IS_SINCOS = True
DATA_PATH = "Downloads/R3J-AGNN/data/dataset.pt"
OUTPUT_DIR = '/Downloads/R3J-AGNN/output/KFold'


# 角度转换
def ensure_angles_in_degrees(pred, is_sincos=PRED_IS_SINCOS):
    if is_sincos:
        sin_vals = pred[:, 0::2]  # [N, 3]
        cos_vals = pred[:, 1::2]  # [N, 3]

        # # 可选：检查 sincos 范围是否符合预期（调试用）
        # assert (sin_vals >= 0).all() and (sin_vals <= 1).all(), "sin values out of [0,1]!"
        # assert (cos_vals >= -1).all() and (cos_vals <= 1).all(), "cos values out of [-1,1]!"

        angles_rad = torch.atan2(sin_vals, cos_vals)  # range: [0, π]
        angles_deg = torch.rad2deg(angles_rad)  # range: [0, 180]

    else:
        angles_deg = pred.view(-1, 3)
        angles_deg = torch.remainder(angles_deg, 360.0)

    return angles_deg


# 评估指标函
def calculate_metrics(pred, target_angles, pred_is_sincos=PRED_IS_SINCOS):
    with torch.no_grad():
        # 将模型输出转换为角度（假设 pred 是 [sin1, cos1, sin2, cos2, sin3, cos3]）
        pred_angles = ensure_angles_in_degrees(pred, is_sincos=pred_is_sincos)  # [N, 3]

        # 有效样本掩码（避免全零无效样本）
        valid_mask = (target_angles.abs().sum(dim=1) > 1e-6)
        if valid_mask.sum() == 0:
            return {
                'total_mse': 0.0,
                'total_mae': 0.0,
                'angle1_mse': 0.0, 'angle1_mae': 0.0,
                'angle2_mse': 0.0, 'angle2_mae': 0.0,
                'angle3_mse': 0.0, 'angle3_mae': 0.0,
            }

        pred_valid = pred_angles[valid_mask]  # [M, 3] —— 顺序：[α, β, γ]
        target_valid = target_angles[valid_mask]  # [M, 3] —— 顺序：[α, β, γ]（原始顺序）

        # 直接比较相同位置的角度
        diff = torch.abs(pred_valid - target_valid)
        mse = (diff ** 2).mean()
        mae = diff.mean()

        metrics = {
            'total_mse': mse.item(),
            'total_mae': mae.item(),
        }
        for i in range(3):
            angle_diff_i = diff[:, i]
            metrics[f'angle{i + 1}_mse'] = (angle_diff_i ** 2).mean().item()
            metrics[f'angle{i + 1}_mae'] = angle_diff_i.mean().item()
        return metrics


# 计算准确率（在阈值范围内）
def calculate_accuracy(pred, target_angles, threshold=THRESHOLD, pred_is_sincos=PRED_IS_SINCOS):
    with torch.no_grad():
        pred_angles = ensure_angles_in_degrees(pred, is_sincos=pred_is_sincos)  # [N, 3]

        valid_mask = (target_angles.abs().sum(dim=1) > 1e-6)
        if valid_mask.sum() == 0:
            return {k: 0.0 for k in ['angle1_acc', 'angle2_acc', 'angle3_acc', 'total_acc']}

        pred_valid = pred_angles[valid_mask]  # [M, 3] —— [α, β, γ]
        target_valid = target_angles[valid_mask]  # [M, 3] —— [α, β, γ]（原始顺序）

        # 直接逐位置比较
        diff = torch.abs(pred_valid - target_valid)
        within_threshold = diff <= threshold
        angle_accs = within_threshold.float().mean(dim=0)
        all_correct = within_threshold.all(dim=1)
        total_acc = all_correct.float().mean()

        # print("pred:", pred)
        # print("valid_mask:", valid_mask)
        # print("pred_valid:", pred_valid)
        # print("target_valid:", target_valid)
        # print("threshold:", threshold)
        # print("diff:", diff)
        # print("within_threshold:", within_threshold)
        # print("angle_accs:", angle_accs)
        # print("all_correct:", all_correct)
        # print("total_acc:", total_acc)
        # print("\n")

        return {
            'angle1_acc': angle_accs[0].item(),
            'angle2_acc': angle_accs[1].item(),
            'angle3_acc': angle_accs[2].item(),
            'total_acc': total_acc.item()
        }


# 损失函数
class CombinedAngleLoss(nn.Module):
    def __init__(
            self,
            loss_type='mse',
            weight_sum=5.0,  # 角度和惩罚权重
            weight_rank=0.5,  # 排序/差分损失权重
            angle_weights=None,
            sum_norm_scale=360.0,
            rank_norm_scale=180.0
    ):
        super().__init__()
        if loss_type not in ('mse', 'l1'):
            raise ValueError("loss_type must be 'mse' or 'l1'")

        self.loss_type = loss_type
        self.weight_sum = weight_sum
        self.weight_rank = weight_rank
        self.sum_norm_scale = float(sum_norm_scale)
        self.rank_norm_scale = float(rank_norm_scale)  # 归一化分母

        # 建议保持均等权重，避免干扰排序学习
        if angle_weights is None:
            self.angle_weights = torch.tensor([1.0, 1.0, 1.0])
        else:
            self.angle_weights = torch.tensor(angle_weights, dtype=torch.float32)

    def forward(self, pred_sincos, target_angles, mask=None):
        device = pred_sincos.device
        self.angle_weights = self.angle_weights.to(device)

        N = pred_sincos.size(0)
        if mask is None:
            mask = torch.ones(N, dtype=torch.bool, device=device)

        # 1. 数据准备
        target_sincos = angles_to_sincos(target_angles)

        pred_valid = pred_sincos[mask]
        target_valid = target_sincos[mask]
        target_angles_valid = target_angles[mask]

        M = pred_valid.size(0)
        if M == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {'angle1_loss': zero, 'rank_loss': zero, 'sum_penalty': zero}

        # ==========================================
        # 2. 基础 Loss (MSE/L1 of Sin/Cos)
        # ==========================================
        # Range: [0, ~4], usually < 1
        pred_reshaped = pred_valid.view(M, 3, 2)
        target_reshaped = target_valid.view(M, 3, 2)

        if self.loss_type == 'mse':
            raw_loss = F.mse_loss(pred_reshaped, target_reshaped, reduction='none').mean(dim=2)
        else:
            raw_loss = F.l1_loss(pred_reshaped, target_reshaped, reduction='none').mean(dim=2)

        weights = self.angle_weights.view(1, 3)
        weighted_loss = raw_loss * weights
        base_loss = weighted_loss.sum() / (weights.sum() * M)

        # ==========================================
        # 3. 归一化的排序/差分损失
        # ==========================================
        pred_sin = pred_valid[:, ::2]
        pred_cos = pred_valid[:, 1::2]
        pred_rad = torch.atan2(pred_sin, pred_cos)
        pred_deg = torch.rad2deg(pred_rad)  # (M, 3), range [0, 180]

        # 计算循环差分: (A1-A2), (A2-A3), (A3-A1)
        # 原始差分范围: [-180, 180]
        target_diffs = target_angles_valid - torch.roll(target_angles_valid, shifts=-1, dims=1)
        pred_diffs = pred_deg - torch.roll(pred_deg, shifts=-1, dims=1)

        # [归一化到 [-1, 1]
        # MSE 的量级就会回到 [0, 1] 左右，与 base_loss 一致
        target_diffs_norm = target_diffs / self.rank_norm_scale
        pred_diffs_norm = pred_diffs / self.rank_norm_scale

        if self.loss_type == 'mse':
            rank_loss = F.mse_loss(pred_diffs_norm, target_diffs_norm)
        else:
            rank_loss = F.l1_loss(pred_diffs_norm, target_diffs_norm)

        # ==========================================
        # 4. 归一化的角度和约束
        # ==========================================
        sum_pred = pred_deg.sum(dim=1)
        sum_error = (sum_pred - 360.0) / self.sum_norm_scale  # Range ~ [-1, 1]

        if self.loss_type == 'mse':
            sum_penalty = (sum_error ** 2).mean()
        else:
            sum_penalty = sum_error.abs().mean()

        # 5. 总 Loss
        total_loss = base_loss + self.weight_rank * rank_loss + self.weight_sum * sum_penalty

        # 日志
        per_angle_display = weighted_loss.mean(dim=0)

        return total_loss, {
            'angle1_loss': per_angle_display[0],
            'angle2_loss': per_angle_display[1],
            'angle3_loss': per_angle_display[2],
            'rank_loss': rank_loss,
            'sum_penalty': sum_penalty,
        }


# 训练函数
def train_epoch(model, loader, criterion, optimizer, device, threshold=THRESHOLD, lambda_l2=0.000001):
    model.train()

    # 累积器
    total_valid_samples = 0

    # 指标累积器
    epoch_metrics = {
        'loss_sum': 0.0,
        'mse_sum': 0.0,
        'mae_sum': 0.0,
        'rmsd_sum': 0.0,
        'correct_count': 0.0,
        'angle1_correct': 0.0,
        'angle2_correct': 0.0,
        'angle3_correct': 0.0,
    }

    # 角度详细指标累积
    angle_metrics_sum = {
        f'angle{i + 1}': {'mse': 0.0, 'mae': 0.0}
        for i in range(3)
    }

    loss_components_sum = {'angle1': 0.0, 'angle2': 0.0, 'angle3': 0.0}

    for batch in tqdm(loader, desc='Training', leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # === 1. 前向传播 ===
        pred = model(batch)  # (N, 6)
        target = batch.y  # (N, 3)
        mask = (target != 0).any(dim=1)  # (N,)

        # 计算有效节点数量
        current_valid_num = mask.sum().item()
        if current_valid_num == 0: continue  # 跳过全 Padding 的 Batch

        # === 2. 计算 Loss ===
        loss, losses = criterion(pred, target, mask=mask)

        # L2 正则
        micro_l2 = sum(p.pow(2).mean() for n, p in model.named_parameters() if 'micro' in n)
        loss = loss + lambda_l2 * micro_l2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # === 3. 数据筛选与指标计算 ===
        # 筛选有效数据
        pred_valid = pred[mask]
        target_valid = target[mask]

        # 计算指标
        batch_metrics = calculate_metrics(pred_valid, target_valid, pred_is_sincos=PRED_IS_SINCOS)
        batch_acc = calculate_accuracy(pred_valid, target_valid, threshold, pred_is_sincos=PRED_IS_SINCOS)

        # 计算 RMSD (compute_rmsd 内部如果已处理 mask 则这里传入 mask=None)
        # 因为 pred_valid 已经是筛选过的，所以这里 mask=None
        rmsd_per_sample = compute_rmsd(pred_valid, target_valid, mask=None)

        # === 4. 累积逻辑 ===
        total_valid_samples += current_valid_num

        epoch_metrics['loss_sum'] += loss.item() * current_valid_num
        epoch_metrics['mse_sum'] += batch_metrics['total_mse'] * current_valid_num
        epoch_metrics['mae_sum'] += batch_metrics['total_mae'] * current_valid_num
        epoch_metrics['rmsd_sum'] += rmsd_per_sample.sum().item()

        epoch_metrics['correct_count'] += batch_acc['total_acc'] * current_valid_num
        epoch_metrics['angle1_correct'] += batch_acc['angle1_acc'] * current_valid_num
        epoch_metrics['angle2_correct'] += batch_acc['angle2_acc'] * current_valid_num
        epoch_metrics['angle3_correct'] += batch_acc['angle3_acc'] * current_valid_num

        for i in range(3):
            key = f'angle{i + 1}'
            angle_metrics_sum[key]['mse'] += batch_metrics[f'{key}_mse'] * current_valid_num
            angle_metrics_sum[key]['mae'] += batch_metrics[f'{key}_mae'] * current_valid_num

            # 兼容 loss key
            loss_key = f'{key}_loss'
            if loss_key not in losses and f'cos{i + 1}_loss' in losses: loss_key = f'cos{i + 1}_loss'
            if loss_key in losses:
                loss_components_sum[key] += losses[loss_key].item() * current_valid_num

    # === 5. 计算最终平均值 ===
    if total_valid_samples == 0: total_valid_samples = 1

    avg_metrics = {
        'loss': epoch_metrics['loss_sum'] / total_valid_samples,
        'mse': epoch_metrics['mse_sum'] / total_valid_samples,
        'mae': epoch_metrics['mae_sum'] / total_valid_samples,
        'rmsd': epoch_metrics['rmsd_sum'] / total_valid_samples,
        'total_acc': epoch_metrics['correct_count'] / total_valid_samples,
        'angle1_acc': epoch_metrics['angle1_correct'] / total_valid_samples,
        'angle2_acc': epoch_metrics['angle2_correct'] / total_valid_samples,
        'angle3_acc': epoch_metrics['angle3_correct'] / total_valid_samples,
    }

    final_angle_metrics = {}
    for key, vals in angle_metrics_sum.items():
        final_angle_metrics[key] = {
            'mse': vals['mse'] / total_valid_samples,
            'mae': vals['mae'] / total_valid_samples
        }

    avg_loss_components = {k: v / total_valid_samples for k, v in loss_components_sum.items()}
    avg_metrics['angle_metrics'] = final_angle_metrics
    avg_metrics['loss_components'] = avg_loss_components

    return avg_metrics


# 验证函数
def validate(model, loader, criterion, device, threshold=THRESHOLD):
    model.eval()

    total_valid_samples = 0

    epoch_metrics = {
        'loss_sum': 0.0,
        'mse_sum': 0.0,
        'mae_sum': 0.0,
        'rmsd_sum': 0.0,
        'correct_count': 0.0,
        'angle1_correct': 0.0,
        'angle2_correct': 0.0,
        'angle3_correct': 0.0,
    }

    angle_metrics_sum = {f'angle{i + 1}': {'mse': 0.0, 'mae': 0.0} for i in range(3)}
    loss_components_sum = {'angle1': 0.0, 'angle2': 0.0, 'angle3': 0.0}

    for batch in tqdm(loader, desc='Validating', leave=False):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)
            target = batch.y
            mask = (target != 0).any(dim=1)

            current_valid_num = mask.sum().item()
            if current_valid_num == 0: continue

            # === 关键修正点: 直接传递 Tensor ===
            loss, losses = criterion(pred, target, mask=mask)

            # 筛选有效数据
            pred_valid = pred[mask]
            target_valid = target[mask]

            # 计算指标
            batch_metrics = calculate_metrics(pred_valid, target_valid, pred_is_sincos=PRED_IS_SINCOS)
            batch_acc = calculate_accuracy(pred_valid, target_valid, threshold, pred_is_sincos=PRED_IS_SINCOS)
            rmsd_per_sample = compute_rmsd(pred_valid, target_valid, mask=None)

            # 累积
            total_valid_samples += current_valid_num
            epoch_metrics['loss_sum'] += loss.item() * current_valid_num
            epoch_metrics['mse_sum'] += batch_metrics['total_mse'] * current_valid_num
            epoch_metrics['mae_sum'] += batch_metrics['total_mae'] * current_valid_num
            epoch_metrics['rmsd_sum'] += rmsd_per_sample.sum().item()

            epoch_metrics['correct_count'] += batch_acc['total_acc'] * current_valid_num
            epoch_metrics['angle1_correct'] += batch_acc['angle1_acc'] * current_valid_num
            epoch_metrics['angle2_correct'] += batch_acc['angle2_acc'] * current_valid_num
            epoch_metrics['angle3_correct'] += batch_acc['angle3_acc'] * current_valid_num

            for i in range(3):
                key = f'angle{i + 1}'
                angle_metrics_sum[key]['mse'] += batch_metrics[f'{key}_mse'] * current_valid_num
                angle_metrics_sum[key]['mae'] += batch_metrics[f'{key}_mae'] * current_valid_num

                loss_key = f'{key}_loss'
                if loss_key not in losses and f'cos{i + 1}_loss' in losses: loss_key = f'cos{i + 1}_loss'
                if loss_key in losses:
                    loss_components_sum[key] += losses[loss_key].item() * current_valid_num

    if total_valid_samples == 0: total_valid_samples = 1

    avg_metrics = {
        'loss': epoch_metrics['loss_sum'] / total_valid_samples,
        'mse': epoch_metrics['mse_sum'] / total_valid_samples,
        'mae': epoch_metrics['mae_sum'] / total_valid_samples,
        'rmsd': epoch_metrics['rmsd_sum'] / total_valid_samples,
        'total_acc': epoch_metrics['correct_count'] / total_valid_samples,
        'angle1_acc': epoch_metrics['angle1_correct'] / total_valid_samples,
        'angle2_acc': epoch_metrics['angle2_correct'] / total_valid_samples,
        'angle3_acc': epoch_metrics['angle3_correct'] / total_valid_samples,
    }

    final_angle_metrics = {}
    for key, vals in angle_metrics_sum.items():
        final_angle_metrics[key] = {
            'mse': vals['mse'] / total_valid_samples,
            'mae': vals['mae'] / total_valid_samples
        }

    avg_loss_components = {k: v / total_valid_samples for k, v in loss_components_sum.items()}
    avg_metrics['angle_metrics'] = final_angle_metrics
    avg_metrics['loss_components'] = avg_loss_components

    return avg_metrics


# 主训练函数
def train_model(train_loader, test_loader, model, optimizer, scheduler, criterion, save_path, device, num_epochs=100,
                patience=60, threshold=THRESHOLD):
    model = model.to(device)
    criterion = criterion
    best_test_acc = 0.0
    best_model_state = None
    no_improvement_epochs = 0
    best_test_rmsd = float('inf')

    # 初始化历史记录（全部使用 angle 命名）
    history = {
        # 整体指标
        'train_loss': [], 'train_acc': [], 'train_mse': [], 'train_mae': [], 'train_rmsd': [],
        'test_loss': [], 'test_acc': [], 'test_mse': [], 'test_mae': [], 'test_rmsd': [],

        # 每个角度的指标（训练）
        'train_angle1_acc': [], 'train_angle1_mae': [], 'train_angle1_mse': [],
        'train_angle2_acc': [], 'train_angle2_mae': [], 'train_angle2_mse': [],
        'train_angle3_acc': [], 'train_angle3_mae': [], 'train_angle3_mse': [],

        # 每个角度的指标（测试）
        'test_angle1_acc': [], 'test_angle1_mae': [], 'test_angle1_mse': [],
        'test_angle2_acc': [], 'test_angle2_mae': [], 'test_angle2_mse': [],
        'test_angle3_acc': [], 'test_angle3_mae': [], 'test_angle3_mse': [],
    }

    def save_checkpoint(state, is_best, filename):
        torch.save(state, filename)
        if is_best:
            best_filename = filename.replace('.pth', '_best.pkl')
            torch.save(state, best_filename)

    for epoch in range(num_epochs):
        # 训练和验证
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, threshold=threshold
        )
        test_metrics = validate(
            model, test_loader, criterion, device, threshold=threshold
        )

        # 更新学习率
        scheduler.step(test_metrics['loss'])

        # === 记录整体指标 ===
        for phase, metrics in zip(['train', 'test'], [train_metrics, test_metrics]):
            history[f'{phase}_loss'].append(metrics['loss'])
            history[f'{phase}_acc'].append(metrics['total_acc'])
            history[f'{phase}_mse'].append(metrics['mse'])
            history[f'{phase}_mae'].append(metrics['mae'])
            history[f'{phase}_rmsd'].append(metrics['rmsd'])

        # === 记录每个角度的指标 ===
        for i in range(3):
            angle_key = f'angle{i+1}'
            # 训练集
            history[f'train_{angle_key}_acc'].append(train_metrics[f'{angle_key}_acc'])
            history[f'train_{angle_key}_mae'].append(train_metrics['angle_metrics'][angle_key]['mae'])
            history[f'train_{angle_key}_mse'].append(train_metrics['angle_metrics'][angle_key]['mse'])

            # 测试集
            history[f'test_{angle_key}_acc'].append(test_metrics[f'{angle_key}_acc'])
            history[f'test_{angle_key}_mae'].append(test_metrics['angle_metrics'][angle_key]['mae'])
            history[f'test_{angle_key}_mse'].append(test_metrics['angle_metrics'][angle_key]['mse'])

        # === 早停与模型保存 ===
        current_test_rmsd = test_metrics['rmsd']
        is_best = current_test_rmsd < best_test_rmsd

        if is_best:
            best_test_rmsd = current_test_rmsd
            best_model_state = model.state_dict()
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # 保存 checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_test_rmsd': best_test_rmsd,
            'history': history,
            'threshold': threshold,
        }
        save_checkpoint(checkpoint, is_best, save_path)

        # 打印日志
        # print(f"Epoch {epoch+1}/{num_epochs}\n"
        #       f"Train Loss: {train_metrics['loss']:.4f}, RMSD: {train_metrics['rmsd']:.4f}, MSE: {train_metrics['mse']:.2f}, MAE: {train_metrics['mae']:.2f}, Acc: {train_metrics['total_acc']:.4f}\n"
        #       f"Test  Loss: {test_metrics['loss']:.4f}, RMSD: {test_metrics['rmsd']:.4f}, MSE: {test_metrics['mse']:.2f}, MAE: {test_metrics['mae']:.2f}, Acc: {test_metrics['total_acc']:.4f}")

        # 早停判断
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    print(f"Training finished. Best Test RMSD: {best_test_rmsd:.4f}")
    if torch.cuda.is_available():
        print(f'GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

    return best_model_state, history


def train_kfold_from_file(data_path, base_output_dir, k_folds=5):
    """
    专门针对已有的训练集 pt 文件执行 K 折交叉验证
    """
    os.makedirs(base_output_dir, exist_ok=True)

    # 1. 加载预先准备好的训练数据集
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到训练集文件: {data_path}")

    print(f"[*] Loading training set from {data_path}...")
    cv_data = torch.load(data_path)
    print(f"[+] Loaded {len(cv_data)} samples for {k_folds}-Fold CV.")

    # 2. 设置随机种子确保可复现性
    seeds = 1234
    random.seed(seeds)
    np.random.seed(seeds)
    torch.manual_seed(seeds)

    # 3. 定义 K-Fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seeds)
    criterion = CombinedAngleLoss(loss_type='mse').to(device)

    # 4. 开始循环
    for fold, (train_idx, val_idx) in enumerate(kf.split(cv_data)):
        print(f"\n{'=' * 20} Starting Fold {fold + 1}/{k_folds} {'=' * 20}")

        # 根据索引提取本折的训练子集和验证子集
        train_set = [cv_data[i] for i in train_idx]
        val_set = [cv_data[i] for i in val_idx]

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Fold {fold + 1} - Train size: {len(train_set)}, Val size: {len(val_set)}")

        # 5. 初始化模型架构
        model = DualGraphRNAModel(
            micro_in_channels=5,
            macro_node_dim=34,
            macro_edge_dim=11,
            hidden_dim=200,
            dropout=0.50,
            num_gat_layers=5,
            output_dim=6,
            micro_num_layers=3,
            use_edge_features=True,
            use_transformer=False,
            use_global_attn=False,
        ).to(device)

        # 6. 配置优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        # 7. 路径与训练
        save_path = os.path.join(base_output_dir, f"fold_{fold + 1}_best_model.pkl")

        best_epoch, history = train_model(
            train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            save_path,
            device
        )


