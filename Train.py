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

from model import DualGraphRNAModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 10.0
BATCH_SIZE = 1 
NUM_EPOCHS = 100
PATIENCE = 60
PRED_IS_SINCOS = True
DATA_PATH = "Downloads/R3J-AGNN/data/dataset.pt"
OUTPUT_DIR = '/Downloads/R3J-AGNN/output/KFold'


def train_epoch(model, loader, criterion, optimizer, device, threshold=THRESHOLD, lambda_l2=0.000001):
    model.train()

    # Accumulators
    total_valid_samples = 0

    # Metric Accumulators
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

    # Detailed angle metric accumulators
    angle_metrics_sum = {
        f'angle{i + 1}': {'mse': 0.0, 'mae': 0.0}
        for i in range(3)
    }

    loss_components_sum = {'angle1': 0.0, 'angle2': 0.0, 'angle3': 0.0}

    for batch in tqdm(loader, desc='Training', leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # === 1. Forward Pass ===
        pred = model(batch)  # (N, 6)
        target = batch.y  # (N, 3)
        mask = (target != 0).any(dim=1)  # (N,)

        # Count valid nodes
        current_valid_num = mask.sum().item()
        if current_valid_num == 0: continue  # Skip fully padded batches

        # === 2. Calculate Loss ===
        loss, losses = criterion(pred, target, mask=mask)

        # L2 Regularization
        micro_l2 = sum(p.pow(2).mean() for n, p in model.named_parameters() if 'micro' in n)
        loss = loss + lambda_l2 * micro_l2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # === 3. Data Filtering & Metric Calculation ===
        # Filter valid data
        pred_valid = pred[mask]
        target_valid = target[mask]

        # Calculate metrics
        batch_metrics = calculate_metrics(pred_valid, target_valid, pred_is_sincos=PRED_IS_SINCOS)
        batch_acc = calculate_accuracy(pred_valid, target_valid, threshold, pred_is_sincos=PRED_IS_SINCOS)

        # Calculate RMSD (compute_rmsd handles mask internally if needed, but here pred_valid is already filtered)
        # So we pass mask=None
        rmsd_per_sample = compute_rmsd(pred_valid, target_valid, mask=None)

        # === 4. Accumulate Logic ===
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

            # Compatible loss key
            loss_key = f'{key}_loss'
            if loss_key not in losses and f'cos{i + 1}_loss' in losses: loss_key = f'cos{i + 1}_loss'
            if loss_key in losses:
                loss_components_sum[key] += losses[loss_key].item() * current_valid_num

    # === 5. Calculate Final Averages ===
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

            # === Critical Fix: Pass Tensor Directly ===
            loss, losses = criterion(pred, target, mask=mask)

            # Filter valid data
            pred_valid = pred[mask]
            target_valid = target[mask]

            # Calculate metrics
            batch_metrics = calculate_metrics(pred_valid, target_valid, pred_is_sincos=PRED_IS_SINCOS)
            batch_acc = calculate_accuracy(pred_valid, target_valid, threshold, pred_is_sincos=PRED_IS_SINCOS)
            rmsd_per_sample = compute_rmsd(pred_valid, target_valid, mask=None)

            # Accumulate
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


# ==========================================
# Main Training Function
# ==========================================
def train_model(train_loader, test_loader, model, optimizer, scheduler, criterion, save_path, device, num_epochs=100,
                patience=60, threshold=THRESHOLD):
    model = model.to(device)
    criterion = criterion
    best_test_acc = 0.0
    best_model_state = None
    no_improvement_epochs = 0
    best_test_rmsd = float('inf')

    # Initialize history (all named using 'angle')
    history = {
        # Overall metrics
        'train_loss': [], 'train_acc': [], 'train_mse': [], 'train_mae': [], 'train_rmsd': [],
        'test_loss': [], 'test_acc': [], 'test_mse': [], 'test_mae': [], 'test_rmsd': [],

        # Metrics per angle (Train)
        'train_angle1_acc': [], 'train_angle1_mae': [], 'train_angle1_mse': [],
        'train_angle2_acc': [], 'train_angle2_mae': [], 'train_angle2_mse': [],
        'train_angle3_acc': [], 'train_angle3_mae': [], 'train_angle3_mse': [],

        # Metrics per angle (Test)
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
        # Training and Validation
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, threshold=threshold
        )
        test_metrics = validate(
            model, test_loader, criterion, device, threshold=threshold
        )

        # Update learning rate
        scheduler.step(test_metrics['loss'])

        # === Record overall metrics ===
        for phase, metrics in zip(['train', 'test'], [train_metrics, test_metrics]):
            history[f'{phase}_loss'].append(metrics['loss'])
            history[f'{phase}_acc'].append(metrics['total_acc'])
            history[f'{phase}_mse'].append(metrics['mse'])
            history[f'{phase}_mae'].append(metrics['mae'])
            history[f'{phase}_rmsd'].append(metrics['rmsd'])

        # === Record metrics per angle ===
        for i in range(3):
            angle_key = f'angle{i+1}'
            # Training set
            history[f'train_{angle_key}_acc'].append(train_metrics[f'{angle_key}_acc'])
            history[f'train_{angle_key}_mae'].append(train_metrics['angle_metrics'][angle_key]['mae'])
            history[f'train_{angle_key}_mse'].append(train_metrics['angle_metrics'][angle_key]['mse'])

            # Test set
            history[f'test_{angle_key}_acc'].append(test_metrics[f'{angle_key}_acc'])
            history[f'test_{angle_key}_mae'].append(test_metrics['angle_metrics'][angle_key]['mae'])
            history[f'test_{angle_key}_mse'].append(test_metrics['angle_metrics'][angle_key]['mse'])

        # === Early stopping & Model saving ===
        current_test_rmsd = test_metrics['rmsd']
        is_best = current_test_rmsd < best_test_rmsd

        if is_best:
            best_test_rmsd = current_test_rmsd
            best_model_state = model.state_dict()
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # Save checkpoint
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

        # Print logs (Optional)
        # print(f"Epoch {epoch+1}/{num_epochs}\n"
        #       f"Train Loss: {train_metrics['loss']:.4f}, RMSD: {train_metrics['rmsd']:.4f}, MSE: {train_metrics['mse']:.2f}, MAE: {train_metrics['mae']:.2f}, Acc: {train_metrics['total_acc']:.4f}\n"
        #       f"Test  Loss: {test_metrics['loss']:.4f}, RMSD: {test_metrics['rmsd']:.4f}, MSE: {test_metrics['mse']:.2f}, MAE: {test_metrics['mae']:.2f}, Acc: {test_metrics['total_acc']:.4f}")

        # Check early stopping
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    print(f"Training finished. Best Test RMSD: {best_test_rmsd:.4f}")
    if torch.cuda.is_available():
        print(f'GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

    return best_model_state, history


def train_kfold(data_path, base_output_dir, k_folds=5):
    """
    Perform K-Fold Cross Validation specifically using an existing .pt training set file.
    """
    os.makedirs(base_output_dir, exist_ok=True)

    # 1. Load pre-prepared training dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training set file not found: {data_path}")

    print(f"[*] Loading training set from {data_path}...")
    cv_data = torch.load(data_path)
    print(f"[+] Loaded {len(cv_data)} samples for {k_folds}-Fold CV.")

    # 2. Set random seeds for reproducibility
    seeds = 1234
    random.seed(seeds)
    np.random.seed(seeds)
    torch.manual_seed(seeds)

    # 3. Define K-Fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seeds)
    criterion = CombinedAngleLoss(loss_type='mse').to(device)

    # 4. Start loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(cv_data)):
        print(f"\n{'=' * 20} Starting Fold {fold + 1}/{k_folds} {'=' * 20}")

        # Extract train/val subsets for current fold based on indices
        train_set = [cv_data[i] for i in train_idx]
        val_set = [cv_data[i] for i in val_idx]

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Fold {fold + 1} - Train size: {len(train_set)}, Val size: {len(val_set)}")

        # 5. Initialize model architecture
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

        # 6. Configure optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        # 7. Paths & Training
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
