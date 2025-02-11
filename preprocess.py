import os
import cv2
import torch
import numpy as np
import json
import random

# 設定原始 HMDB51 影片資料夾與預處理後儲存的目錄
DATASET_DIR = "hmdb51/"
OUTPUT_DIR = "preprocessed_data_pt/"
NUM_FRAMES = 32  # 固定取樣的幀數
FRAME_SIZE = (224, 224)  # 影像尺寸
SPLIT_RATIO = 0.8  # 訓練/測試集拆分比例
SEED = 42  # 設定隨機種子，確保每次拆分結果相同

# 若預處理儲存目錄不存在則建立
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video(video_path, num_frames=NUM_FRAMES):
    """ 讀取影片並轉換為固定幀數的張量 """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"⚠️ 警告：影片 {video_path} 無法讀取，跳過")
            cap.release()
            return None

        # 均勻取樣 `num_frames` 幀
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉為 RGB
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)  # 縮放影像
            frames.append(frame)

        cap.release()

        # **改進補幀策略：線性插值**
        if len(frames) < num_frames:
            print(f"⚠️ 影片 {video_path} 幀數不足 ({len(frames)}/{num_frames})，補齊中...")
            idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in idx]

        frames = np.array(frames, dtype=np.float16) / 255.0  # 轉換為 float16 並正規化
        return torch.from_numpy(frames)  # shape: (num_frames, 224, 224, 3)

    except Exception as e:
        print(f"❌ 影片 {video_path} 處理失敗: {e}")
        return None

# 設定隨機種子，確保每次拆分一致
random.seed(SEED)

# 建立 class_to_idx 字典
class_to_idx = {}

# 處理所有影片
for class_index, class_name in enumerate(sorted(os.listdir(DATASET_DIR))):  # 確保類別索引固定
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    # **加入到 class_to_idx**
    class_to_idx[class_name] = class_index

    # 取得所有該類別的影片
    video_files = [f for f in os.listdir(class_dir) if f.endswith(".avi")]
    random.shuffle(video_files)  # 打亂順序，確保分割隨機性

    # 計算訓練/測試集數量
    num_train = int(len(video_files) * SPLIT_RATIO)
    train_videos = video_files[:num_train]
    test_videos = video_files[num_train:]

    # 建立訓練 & 測試的儲存資料夾
    train_class_dir = os.path.join(OUTPUT_DIR, "train", class_name)
    test_class_dir = os.path.join(OUTPUT_DIR, "test", class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # **處理訓練影片**
    for video_file in train_videos:
        video_path = os.path.join(class_dir, video_file)
        frames_tensor = process_video(video_path)

        if frames_tensor is not None:
            save_path = os.path.join(train_class_dir, video_file.replace(".avi", ".pt"))
            torch.save(frames_tensor, save_path)
            print(f"✅ 儲存 (訓練): {save_path}")

    # **處理測試影片**
    for video_file in test_videos:
        video_path = os.path.join(class_dir, video_file)
        frames_tensor = process_video(video_path)

        if frames_tensor is not None:
            save_path = os.path.join(test_class_dir, video_file.replace(".avi", ".pt"))
            torch.save(frames_tensor, save_path)
            print(f"✅ 儲存 (測試): {save_path}")

# **將 class_to_idx 儲存為 JSON**
class_to_idx_path = os.path.join(OUTPUT_DIR, "class_to_idx.json")
with open(class_to_idx_path, "w") as f:
    json.dump(class_to_idx, f, indent=4)

print(f"📄 類別索引已儲存至: {class_to_idx_path}")
