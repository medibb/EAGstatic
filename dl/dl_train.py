"""
EAG-GRF Deep Learning Training

Usage:
    # Task A: 동작 분류
    python3 dl_train.py --task classify_action --model cnn --epochs 50

    # Task B: 체중부하 분류 (crutch 세션만)
    python3 dl_train.py --task classify_load --model cnn --epochs 50

    # Task C: GRF 재구성
    python3 dl_train.py --task regress_grf --model regressor --epochs 100
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, f1_score)
from pathlib import Path

from dl_dataset import build_dataset, create_subject_splits, SESSION_TYPE_NAMES
from dl_models import EAG1DCNN, EAGLSTM, EAGRegressor, count_params


def get_model(model_name: str, task: str, n_classes: int = 3):
    if model_name == 'cnn':
        return EAG1DCNN(n_classes=n_classes)
    elif model_name == 'lstm':
        return EAGLSTM(n_classes=n_classes)
    elif model_name == 'regressor':
        return EAGRegressor()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_classifier(model, train_loader, val_loader, test_loader,
                     n_classes, class_names, epochs, lr, save_dir, task_name):
    """분류 모델 학습 + 평가."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Class weights (imbalance 대응)
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    counts = np.bincount(all_labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    class_weights = torch.FloatTensor(weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    history = {'epoch': [], 'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        train_loss /= total
        train_acc = correct / total

        # --- Val ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)
                val_preds.extend(out.argmax(1).numpy())
                val_true.extend(y.numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_true, val_preds, average='macro')

        scheduler.step(val_loss)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['train_acc'].append(round(train_acc, 4))
        history['val_acc'].append(round(val_acc, 4))
        history['val_f1'].append(round(val_f1, 4))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_dir / 'model_best.pt')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={train_loss:.4f}/{val_loss:.4f} "
                  f"acc={train_acc:.3f}/{val_acc:.3f} "
                  f"F1={val_f1:.3f}")

    # --- Test (best model) ---
    model.load_state_dict(torch.load(save_dir / 'model_best.pt', weights_only=True))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            test_preds.extend(out.argmax(1).numpy())
            test_true.extend(y.numpy())

    test_acc = np.mean(np.array(test_preds) == np.array(test_true))
    test_f1 = f1_score(test_true, test_preds, average='macro')

    print(f"\n=== Test Results ({task_name}) ===")
    print(f"  Accuracy: {test_acc:.3f}")
    print(f"  F1 (macro): {test_f1:.3f}")
    print(classification_report(test_true, test_preds,
                                target_names=class_names[:n_classes]))

    # Confusion matrix
    cm = confusion_matrix(test_true, test_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names[:n_classes])
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'{task_name} — Test Acc={test_acc:.3f}, F1={test_f1:.3f}')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=150)
    plt.close()

    # Learning curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['epoch'], history['train_loss'], label='Train')
    ax1.plot(history['epoch'], history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['epoch'], history['train_acc'], label='Train Acc')
    ax2.plot(history['epoch'], history['val_acc'], label='Val Acc')
    ax2.plot(history['epoch'], history['val_f1'], label='Val F1', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Accuracy & F1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{task_name} Learning Curve', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'learning_curve.png', dpi=150)
    plt.close()

    # Save metrics
    import pandas as pd
    pd.DataFrame(history).to_csv(save_dir / 'metrics.csv', index=False)

    return test_acc, test_f1


def train_regressor(model, train_loader, val_loader, test_loader,
                    epochs, lr, save_dir):
    """GRF 재구성 학습 + 평가."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss, total = 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            total += x.size(0)
        train_loss /= total

        model.eval()
        val_loss, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_total += x.size(0)
        val_loss /= val_total

        scheduler.step(val_loss)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(round(train_loss, 6))
        history['val_loss'].append(round(val_loss, 6))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / 'model_best.pt')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"MSE={train_loss:.6f}/{val_loss:.6f}")

    # --- Test ---
    model.load_state_dict(torch.load(save_dir / 'model_best.pt', weights_only=True))
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            all_preds.append(out.numpy())
            all_true.append(y.numpy())

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_true)

    mse = np.mean((preds - trues) ** 2)
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n=== Test Results (GRF Regression) ===")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²: {r2:.4f}")

    # 예측 vs 실측 plot (첫 3개 샘플)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    t = np.arange(preds.shape[-1]) / 250.0
    for i, ax in enumerate(axes):
        if i >= len(preds):
            break
        ax.plot(t, trues[i, 0], 'b-', linewidth=0.8, label='True L', alpha=0.7)
        ax.plot(t, preds[i, 0], 'b--', linewidth=0.8, label='Pred L')
        ax.plot(t, trues[i, 1], 'r-', linewidth=0.8, label='True R', alpha=0.7)
        ax.plot(t, preds[i, 1], 'r--', linewidth=0.8, label='Pred R')
        ax.set_ylabel('GRF (norm)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'GRF Reconstruction — MSE={mse:.4f}, R²={r2:.3f}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'prediction_plot.png', dpi=150)
    plt.close()

    import pandas as pd
    pd.DataFrame(history).to_csv(save_dir / 'metrics.csv', index=False)

    return mse, r2


def main():
    parser = argparse.ArgumentParser(description='EAG-GRF DL Training')
    parser.add_argument('--task', type=str, default='classify_action',
                        choices=['classify_action', 'classify_load', 'regress_grf'])
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'regressor'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print("=" * 60)

    # Dataset
    dataset = build_dataset(task=args.task)
    train_ds, val_ds, test_ds = create_subject_splits(dataset, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model
    if args.task == 'classify_action':
        n_classes = 3
        class_names = SESSION_TYPE_NAMES
    elif args.task == 'classify_load':
        n_classes = 4
        class_names = ['20%', '50%', '80%', '100%']
    else:
        n_classes = 2  # regression output channels

    model = get_model(args.model, args.task, n_classes)
    print(f"Model params: {count_params(model):,}")

    save_dir = f'result/dl/{args.task}_{args.model}'

    if args.task.startswith('classify'):
        train_classifier(model, train_loader, val_loader, test_loader,
                         n_classes, class_names, args.epochs, args.lr,
                         save_dir, args.task)
    else:
        train_regressor(model, train_loader, val_loader, test_loader,
                        args.epochs, args.lr, save_dir)

    print(f"\n결과 저장: {save_dir}/")


if __name__ == '__main__':
    main()
