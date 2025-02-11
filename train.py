"""
改進的 PyTorch 訓練腳本，適用於 ViViT 模型在 HMDB51 數據集上的訓練。
此版本整合了：
  - 原始數據與數據增強 (augmented) 兩種訓練模式，
  - AMP 自動混合精度 (Automatic Mixed Precision) 與梯度裁剪，
  - CosineAnnealingLR 學習率調度器，
  - checkpoint 載入與儲存，
"""
import os
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import ConcatDataset, Subset
import random
from transformers import get_cosine_schedule_with_warmup
import numpy as np

# 啟用 cudnn 優化
torch.backends.cudnn.benchmark = True

# 從 model.py 載入各模型（請確保模型名稱與你的訓練需求一致）
from model import model1, ViViT_Factorized, ViViT_Factorized_selfAttention, ViViT_Factorized_DotProduct
# 從 dataset.py 載入 HMDB51Dataset（此 Dataset 需支援 mode 參數：raw 或 augmented）
from dataset import HMDB51Dataset

# 建立全域的 AMP GradScaler
scaler = torch.cuda.amp.GradScaler()

def get_model(model_name: str):
    """
    根據模型名稱返回對應的模型實例。
    """
    model_dict = {
        'model1': model1(
            in_channels=3, embed_dim=128, patch_size=16, tubelet_size=2,
            num_heads=8, mlp_dim=512, num_layers=6, num_classes=51,
            num_frames=32, img_size=224
        ),
        'vivit_factorized': ViViT_Factorized(
            in_channels=3, embed_dim=64, patch_size=16, tubelet_size=2,
            num_heads=8, mlp_dim=384, num_layers_spatial=4, num_layers_temporal=4,
            num_classes=51, num_frames=32, img_size=224, droplayer_p=0.1
        ),
        'vivit_factorized_selfattention': ViViT_Factorized_selfAttention(
            in_channels=3, embed_dim=64, patch_size=16, tubelet_size=2,
            num_heads=8, mlp_dim=64 * 4, num_layers=6, num_classes=51,
            num_frames=32, img_size=224, dropout=0.6, droplayer_p=0.2
        ),
        'vivit_factorized_dotproduct': ViViT_Factorized_DotProduct(
            in_channels=3, embed_dim=64, patch_size=16, tubelet_size=2,
            num_heads=8, mlp_dim=64 * 4, num_layers=4, num_classes=51,
            num_frames=32, img_size=224,
        ),
    }
    if model_name not in model_dict:
        raise ValueError(f"❌ 錯誤：不支援的模型 `{model_name}`！") # 直接讓程式停止執行
    return model_dict[model_name]

def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    criterion: nn.Module, # loss function
                    optimizer: optim.Optimizer,
                    scheduler, # 學習率調整器
                    device: torch.device, # cpu、gpu
                    epoch: int, # 當前 epoch 編號
                    accumulation_steps: int = 4) -> (float, float):
    """
    訓練一個 epoch，支援梯度累積，並且只包含合併後的數據集 `train_loader`。
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    optimizer.zero_grad()  # 保證每個 epoch 初始時梯度歸零

    # 訓練單一混合數據集
    print(f"\n🚀 Epoch {epoch} - 訓練混合數據集 ")
    # batch_idx : batch index, (inputs, labels) : (影片, 分類結果)
    # train_loader : DataLoader(train_dataset, batch_size, shuffle=True)
    # 從train_dataset隨機一個 batch (包含 batch_size 個 frame)，並返回索引batch_idx、資料(inputs, labels)
    # tqdm顯示進度條、desc:進度條標題、leave:留下進度條
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)):
        inputs, labels = inputs.to(device), labels.to(device) # 將label input搬移到 GPU 或 CPU
        with torch.cuda.amp.autocast():  # 啟用 AMP 混合精度
            outputs = model(inputs) # forward pass

            # criterion(outputs, labels):計算當前函數損失值、accumulation_steps=4，表示要累積 4 個 batch 的梯度後再更新一次
            loss = criterion(outputs, labels) / accumulation_steps

        # scale(loss)先放大 loss (避免 FP16 數值精度問題)
        # backward不會馬上清除梯度，會累積在model.parameters()
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0: # 只有是accumulation_steps的倍數才更新梯度
            scaler.unscale_(optimizer) #取消scale(loss)的放大

            #限制所有參數的L2 NORM不超過3。向量[X1,X2]則此向量的NORM是(X1^2+X2^2)^1/2
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0) # 梯度裁剪用來避免梯度爆炸

            # 更新梯度
            scaler.step(optimizer)
            scaler.update() # 調整 AMP 縮放因子

            # 清除梯度
            optimizer.zero_grad()

        _, preds = torch.max(outputs, 1) # 沿著維度 1(類別維度)找出最大值(預測的類別)
        correct += preds.eq(labels).sum().item() # 計算預測正確的樣本數
        total += labels.size(0) #取得當前batch的樣本數，累加到total
        total_loss += loss.item() * accumulation_steps  # loss需乘回accumulation_steps

    avg_loss = total_loss / len(train_loader)  # 確保 Loss 平均計算
    accuracy = correct / total * 100.0
    epoch_time = time.time() - start_time
    print(f"\n🔥 Epoch {epoch}: Avg Loss {avg_loss:.4f}, Accuracy {accuracy:.2f}%, Time {epoch_time:.2f} sec")

    scheduler.step()
    return avg_loss, accuracy

def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device) -> (float, float):
    """
    評估模型在測試集上的表現，計算 Loss 與 Accuracy。

    Args:
        model (nn.Module)       : 已訓練好的模型
        test_loader (DataLoader) : 測試集的 DataLoader
        criterion (nn.Module)    : 損失函數 (如 CrossEntropyLoss)
        device (torch.device)    : 運算設備 (CPU/GPU)

    Returns:
        avg_loss (float) : 測試集的平均 Loss
        accuracy (float) : 測試集的準確率 (%)
    """

    model.eval()  # 設置模型為「評估模式」 (Evaluation Mode)，關閉 Dropout/BatchNorm

    total_loss = 0.0  # 累積 Loss
    correct = 0  # 預測正確的樣本數
    total = 0  # 測試集總樣本數

    with torch.no_grad():  # 關閉梯度計算，節省記憶體與計算資源
        for inputs, labels in test_loader:  # 遍歷測試集每個 batch
            inputs, labels = inputs.to(device), labels.to(device)  # 將數據搬移到 GPU / CPU

            outputs = model(inputs)  # Forward Pass
            loss = criterion(outputs, labels)  # 計算 Loss
            total_loss += loss.item()  # 累積 Loss

            _, preds = torch.max(outputs, 1)  # 取得預測的類別 (argmax)
            correct += preds.eq(labels).sum().item()  # 計算正確預測數量
            total += labels.size(0)  # 累積樣本總數

    avg_loss = total_loss / len(test_loader)  # 計算平均 Loss
    accuracy = correct / total * 100.0  # 計算準確率 (%)

    print(f"✅ 測試集: Loss {avg_loss:.4f}, Accuracy {accuracy:.2f}%")  # 🔹 顯示評估結果

    return avg_loss, accuracy  # 🔥 回傳測試集的平均 Loss 與準確率

def get_args():
    parser = argparse.ArgumentParser(description="Train ViViT Model on HMDB51")
    parser.add_argument("--epochs", type=int, default=50, help="訓練 Epoch 數")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch 大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="初始學習率")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="權重衰減 (L2 正則化)")
    parser.add_argument("--train_data", type=str, default="preprocessed_data_pt/train", help="訓練數據集路徑")
    parser.add_argument("--test_data", type=str, default="preprocessed_data_pt/test", help="測試數據集路徑")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="儲存模型的資料夾")
    parser.add_argument("--resume", type=str, default=None, help="載入已訓練的模型 (.pth)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["plateau", "cosine"], help="選擇學習率調度器")
    parser.add_argument("--model_name", type=str, default="vivit_factorized_selfattention",
                        choices=["model1", "vivit_factorized", "vivit_factorized_selfattention", "vivit_factorized_dotproduct"],
                        help="選擇要訓練的模型")
    return parser.parse_args()


def save_model(model: nn.Module, optimizer: optim.Optimizer, epoch: int, save_dir: str = "checkpoints"):
    """儲存模型權重與訓練狀態"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"vivit_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, save_path)
    print(f"✅ 模型已儲存至 {save_path}")

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用裝置: {device}")

    writer = SummaryWriter(log_dir="./logs")

    # 載入數據集：訓練數據分為 raw 與 augmented 兩部分
    print("📂 正在載入訓練數據集...")
    train_dataset_raw = HMDB51Dataset(root_dir=args.train_data, mode="raw")
    train_dataset_aug = HMDB51Dataset(root_dir=args.train_data, mode="augmented")

    # 計算增強數據應該取多少（10%）
    num_aug_samples = int(len(train_dataset_raw) * 0.1)  # 10% 增強數據
    aug_indices = random.sample(range(len(train_dataset_aug)), num_aug_samples) #隨機抽取num_aug_samples個數據
    train_dataset_aug_sampled = Subset(train_dataset_aug, aug_indices) #建立抽取後的子數據

    # 合併數據集（全部原始 + 10% 增強）
    train_dataset_combined = ConcatDataset([train_dataset_raw, train_dataset_aug_sampled])

    # 創建新的 DataLoader
    train_loader = DataLoader(
        train_dataset_combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    print("📂 正在載入測試數據集...")
    test_dataset = HMDB51Dataset(root_dir=args.test_data, mode="raw")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # 🚀 使用 `get_model()` 來載入模型
    print(f"📦 載入模型: {args.model_name}")
    model = get_model(args.model_name).to(device)

    # 定義損失函數與優化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用 Label Smoothing
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #參數、學習率、正規化

    # 設定 Warmup 步數，通常為總 epochs 的 10%
    num_warmup_steps = int(args.epochs * 0.1)

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=args.epochs
        )
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 載入 checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)#載入.pth
        model.load_state_dict(checkpoint["model_state_dict"])#載入模型參數(權重、偏差)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])#載入優化器(學習率、梯度動量)
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ 從 Epoch {start_epoch} 繼續訓練")
    else:
        print("❌ 未找到 checkpoint，從 Epoch 1 開始訓練")

    os.makedirs(args.save_dir, exist_ok=True)#在當前路徑建立save_dir資料夾

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n🚀 開始訓練 Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)

        # 每 5 個 epoch 儲存一次 checkpoint、和評估測試集
        if epoch % 5 == 0 or epoch == args.epochs:
            save_model(model, optimizer, epoch, args.save_dir)
            print("\n📊 評估測試集...")
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n🎉 訓練完成！")
    writer.close()

if __name__ == "__main__":
    main()
