import os
import torch
import cv2
import numpy as np
import argparse
import json
from torchvision import transforms
from tqdm import tqdm
from train import get_model


# 影片預處理函數
def preprocess_video(video_path, frame_count=32, frame_size=(224, 224)):
    """
    讀取 .avi 影片並轉換為 ViViT 可處理的格式 (C, T, H, W)。
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 取得影片的總幀數
    frame_indices = np.linspace(0, total_frames - 1, frame_count).astype(int)  # 均勻取樣 frame_count 幀

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # 跳轉到第 idx 幀
        ret, frame = cap.read()  # 讀取該幀
        if not ret:
            break  # 若讀取失敗，則跳出

        frame = cv2.resize(frame, frame_size)  # 調整影像大小 (224x224)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換 BGR → RGB
        frames.append(frame)  # 存入 frames 陣列

    cap.release()

    while len(frames) < frame_count:
        frames.append(frames[-1])  # 若幀數不足 frame_count，則填充最後一幀

    transform = transforms.Compose([
        transforms.ToTensor(),  # 轉換為 Tensor，(H, W, C) → (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
    ])

    frames = torch.stack([transform(frame) for frame in frames])  # 把frames(C, H, W)堆疊成(T, C, H, W)
    frames = frames.permute(1, 0, 2, 3)  # 轉換為 (C, T, H, W)

    return frames.unsqueeze(0)  # 增加 batch 維度 (1, C, T, H, W)

# 影片分類預測
def predict(model, video_tensor, idx_to_class, top_k=5):
    """
    使用 ViViT 模型進行推理，返回 Top-K 結果 (類別名稱)。
    idx_to_class ex:{0: "brush_hair", 1: "cartwheel", 2: "catch", 3: "chew", ...}

    """
    model.eval()
    with torch.no_grad():  # 關閉梯度計算
        outputs = model(video_tensor)  # 模型輸出 logits
        probs = torch.nn.functional.softmax(outputs, dim=1)  # 轉換為機率(0到1之間)
        top_probs, top_classes = torch.topk(probs, top_k, dim=1)  # 取得前k高的機率與類別

    # 取得模型預測的 Top-K 類別索引 (Tensor)
    top_classes = top_classes.squeeze()  # 移除多餘維度(維度為1 即batch size)，使其變成一維張量
    top_classes_list = top_classes.tolist()  # 轉換為 Python 列表

    # 將類別索引轉換為對應的類別名稱
    top_class_names = []  # 存放類別名稱
    for idx in top_classes_list:
        class_name = idx_to_class.get(idx, "Unknown")  # 透過字典查找類別名稱
        top_class_names.append(class_name)  # 加入結果列表

    return top_class_names, top_probs.squeeze().tolist()

def get_args():
    parser = argparse.ArgumentParser(description="使用 ViViT 模型進行影片分類")
    parser.add_argument("video_path", type=str, help="輸入 AVI 影片的路徑")
    parser.add_argument("model_path", type=str, help="已訓練模型的權重文件 (.pth)")
    parser.add_argument("--model_name", type=str, default="vivit_factorized_dotproduct",
                        choices=["model1", "vivit_factorized", "vivit_factorized_selfattention",
                                 "vivit_factorized_dotproduct"],
                        help="選擇要載入的模型")
    parser.add_argument("--top_k", type=int, default=5, help="要顯示的 Top-K 結果")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="運行裝置 (cpu 或 cuda)")
    parser.add_argument("--strict", action="store_true", help="是否強制匹配 state_dict 鍵值")

    return parser.parse_args()

# 主程式
def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"🚀 運行裝置: {device}")

    # 載入 ViViT 模型
    print(f"📦 載入模型: {args.model_name}")
    model = get_model(args.model_name).to(device)

    # 載入 `.pth` 權重檔案
    print(f"🔄 載入權重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # 確保 `model_state_dict` 被正確載入
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  # 直接載入

    # 讀取 class_to_idx.json 並建立 idx_to_class
    print("📂 讀取類別映射文件...")
    # 讀取 class_to_idx.json 並載入類別對應表
    with open("class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)  # 類別名稱 → 數字索引 (e.g., "brush_hair": 0)

    # 反轉字典，使我們可以透過索引找到對應的類別名稱
    idx_to_class = {}
    for class_name, class_index in class_to_idx.items():
        idx_to_class[class_index] = class_name  # 轉換為 (索引 → 類別名稱)

    # 忽略不匹配的鍵值
    try:
        model.load_state_dict(state_dict, strict=args.strict)
        print("✅ 成功載入模型權重！")
    except RuntimeError as e:
        print(f"⚠️ 警告：載入模型權重時發生錯誤，嘗試 `strict=False` 方式載入...\n{e}")
        model.load_state_dict(state_dict, strict=False)

    # 影片預處理
    print(f"🎬 正在處理影片: {args.video_path}")
    video_tensor = preprocess_video(args.video_path).to(device)

    # 進行推理
    print("🔍 進行影片分類...")
    top_class_names, top_probs = predict(model, video_tensor, idx_to_class, top_k=args.top_k)

    # 顯示結果
    print("\n📊 Top-{} 分類結果：".format(args.top_k))
    for i in range(args.top_k):
        print(f"{i + 1}. 類別: {top_class_names[i]}, 機率: {top_probs[i]:.4f}")


if __name__ == "__main__":
    main()
