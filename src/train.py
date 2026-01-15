"""
Asterixis Detection System - Training Script
모델 학습 및 평가
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from .config import MODEL_DIR, TRAINING_CONFIG, PROCESSED_DIR, LABELS
from .dataset import load_data, get_dataloaders
from .model import AsterixisModel, count_parameters


class Trainer:
    """모델 학습 클래스"""

    def __init__(self, model, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self._print_device_info()

    def _print_device_info(self):
        """디바이스 정보 출력"""
        print(f"\nDevice: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def train_epoch(self, train_loader, criterion, optimizer):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward
            loss.backward()
            optimizer.step()

            # 통계
            total_loss += loss.item() * batch_X.size(0)
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        return total_loss / total, correct / total

    def validate(self, val_loader, criterion):
        """검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item() * batch_X.size(0)
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

    def train(
        self,
        train_loader,
        val_loader,
        class_weights,
        epochs=None,
        learning_rate=None,
        patience=None
    ):
        """전체 학습"""
        epochs = epochs or TRAINING_CONFIG["epochs"]
        learning_rate = learning_rate or TRAINING_CONFIG["learning_rate"]
        patience = patience or TRAINING_CONFIG["early_stopping_patience"]

        # 손실 함수
        criterion = nn.BCELoss(reduction='mean')

        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=TRAINING_CONFIG["weight_decay"]
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f'runs/asterixis_{timestamp}')

        # 학습 기록
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = MODEL_DIR / 'asterixis_best.pt'

        self._print_training_header(epochs, learning_rate, patience)

        for epoch in range(epochs):
            # 학습
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # 검증
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)

            # 기록
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # TensorBoard
            writer.add_scalars('Loss', {
                'train': train_loss,
                'validation': val_loss
            }, epoch)
            writer.add_scalars('Accuracy', {
                'train': train_acc,
                'validation': val_acc
            }, epoch)

            # 출력
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping & model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(
                    epoch, optimizer, val_loss, val_acc, best_model_path
                )
                print(f"  -> Model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        writer.close()

        # 최적 모델 로드
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self._print_training_summary(checkpoint, best_model_path)

        return history, val_preds, val_labels

    def _save_checkpoint(self, epoch, optimizer, val_loss, val_acc, path):
        """체크포인트 저장"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, path)

    def _print_training_header(self, epochs, learning_rate, patience):
        """학습 시작 헤더 출력"""
        print("\n" + "=" * 70)
        print("Training Started")
        print("=" * 70)
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Early stopping patience: {patience}")
        print("=" * 70)

    def _print_training_summary(self, checkpoint, best_model_path):
        """학습 완료 요약 출력"""
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
        print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
        print(f"Model saved: {best_model_path}")
        print("=" * 70)

    def save_final_model(self, path=None):
        """최종 모델 저장 (추론용)"""
        path = path or MODEL_DIR / 'asterixis_final.pt'
        torch.save(self.model.state_dict(), path)
        print(f"Final model saved: {path}")


def plot_training_history(history, save_path=None):
    """학습 히스토리 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_title('Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_title('Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Confusion Matrix 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    label_names = [LABELS[0], LABELS[1]]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")

    plt.show()


def main():
    """메인 학습 함수"""
    print("\n" + "=" * 70)
    print("Asterixis Detection - Model Training")
    print("=" * 70)

    # 데이터 로드
    X, y = load_data()
    if X is None:
        print("No data available. Please run preprocessing first.")
        return

    # DataLoader 생성
    train_loader, val_loader, class_weights = get_dataloaders(X, y)

    # 모델 생성
    model = AsterixisModel()
    print(f"\nModel parameters: {count_parameters(model):,}")

    # 학습
    trainer = Trainer(model)
    history, val_preds, val_labels = trainer.train(
        train_loader, val_loader, class_weights
    )

    # 최종 모델 저장
    trainer.save_final_model()

    # 시각화
    plot_training_history(history, MODEL_DIR / 'training_history.png')
    plot_confusion_matrix(val_labels, val_preds, MODEL_DIR / 'confusion_matrix.png')

    # Classification Report
    label_names = [LABELS[0], LABELS[1]]
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=label_names))


if __name__ == "__main__":
    main()
