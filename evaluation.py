#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ViViT 模型測試腳本，適用於 HMDB51 數據集。
會讀取 `.pth` 模型權重，並在測試數據集上計算 Loss 和 Top-1 準確率。
"""

import torch
import argparse
import json
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from model import model1, ViViT_Factorized, ViViT_Factorized_selfAttention, ViViT_Factorized_DotProduct
from dataset import HMDB51Dataset
from tqdm import tqdm
from train import get_model

def evaluate(model, dataloader, criterion, device):
    """在測試數據集上評估模型準確率"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    if len(dataloader) == 0:  # ✅ 檢查 dataloader 是否為空
        print("⚠️ 測試數據集為空，無法評估！")
        return 0.0, 0.0

    torch.cuda.empty_cache()  # 釋放顯存，避免 OOM

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=True):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            assert outputs.dim() == 2, f"❌ 錯誤：模型輸出 shape 錯誤，預期 (batch_size, num_classes)，但得到 {outputs.shape}"

            loss = criterion(outputs, labels)  # 計算 Loss
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total * 100.0 if total > 0 else 0.0

    torch.cuda.empty_cache()  # 測試後釋放顯存

    return avg_loss, accuracy

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate ViViT Model')
    parser.add_argument('--data_path', type=str, required=True, help='測試數據集的路徑')
    parser.add_argument('--model_path', type=str, required=True, help='訓練好的模型檔案 (.pth)')
    parser.add_argument('--model', type=str, required=True, choices=['model1', 'vivit_factorized',
                                                                     'vivit_factorized_selfattention',
                                                                     'vivit_factorized_dotproduct'],
                        help='選擇模型')
    parser.add_argument('--batch_size', type=int, default=8, help='測試時的 batch size')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='運行裝置')
    parser.add_argument('--class_idx', type=str, default="class_to_idx.json", help="類別索引 JSON 檔案")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device)

    print(f"🚀 運行裝置: {device}")

    # 載入類別索引，確保標籤一致
    if os.path.exists(args.class_idx):
        with open(args.class_idx, "r") as f:
            class_to_idx = json.load(f)
        num_classes = len(class_to_idx)
    else:
        print("⚠️ 找不到 `class_to_idx.json`，使用預設類別數 51！")
        num_classes = 51

    # 加載模型
    print(f"🔄 載入模型 `{args.model}`...")
    model = get_model(args.model, num_classes).to(device)

    # 修正模型載入
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint: #model_state_dict是模型的每層的權重和偏差
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)

    # 定義與 train.py 相同的 loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 加載測試數據
    print("🔄 載入測試數據集...")
    test_dataset = HMDB51Dataset(root_dir=args.data_path, mode="raw")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # 開始評估
    print("🚀 開始測試...")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"✅ 測試集: Loss {test_loss:.4f}, Accuracy {test_accuracy:.2f}%")
