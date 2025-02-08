import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as F

#忽略future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class HMDB51Dataset(Dataset):
    def __init__(self, root_dir, mode="raw"):
        """
        Args:
            root_dir (str): 數據集的根目錄 (`preprocessed_data_pt/train/` 或 `preprocessed_data_pt/test/`)
            mode (str): "raw" 表示不做數據增強, "augmented" 表示使用數據增強
        """
        self.root_dir = root_dir
        self.mode = mode  # 控制是否啟用數據增強

        # **基本變換（適用於所有數據）**
        self.base_transform = transforms.Compose([
            ToPILImage(),  # 將 Tensor 轉換為 PIL 圖像，便於後續處理
            transforms.Resize((224, 224)),  # 將圖像調整為 224x224 的大小
            transforms.ToTensor(),  # 轉換回 PyTorch 的 Tensor 格式
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 進行標準化處理
        ])

        # 定義數據增強轉換，這些轉換會隨機改變圖像的外觀，以提升模型的泛化能力
        self.augmentation_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 隨機裁剪並調整大小
            transforms.RandomHorizontalFlip(p=0.5),  # 以 50% 機率翻轉圖像
            transforms.RandomRotation(degrees=10),  # 隨機旋轉 -10 到 10 度
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05))  # 隨機進行仿射變換，包含旋轉和平移
        ])

        # 讀取數據集內的所有資料夾名稱，並按照字母順序排序 ex:["jump", "run", "walk"]
        self.classes = sorted(os.listdir(root_dir))

        # 創建一個從類別名稱映射到索引的字典，ex:{"jump": 0, "run": 1}
        self.class_to_idx = {}
        for i, cls in enumerate(self.classes): # enumerate同時取得索引及對應值
            self.class_to_idx[cls] = i  # 把類別名稱 cls 對應到索引 i

        # 加載所有數據文件的路徑以及對應的標籤
        self.video_paths, self.labels = self.load_data(root_dir)

    def load_data(self, root_dir):
        video_paths, labels = [], []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)# 組合成完整的類別資料夾路徑
            if not os.path.isdir(class_dir):
                continue

            for video_file in os.listdir(class_dir):
                if video_file.endswith('.pt'):
                    video_paths.append(os.path.join(class_dir, video_file))
                    labels.append(self.class_to_idx[class_name])

        return video_paths, labels

    def __len__(self):
        return len(self.video_paths)

    # 類似陣列A={0,1,2}，A[1]=1 就是使用getitem來return 1
    # ex: dataset用init初始化dataset={0,1,2}，用getitem來取得dataset[1]=1
    def __getitem__(self, idx):
        """
        根據索引返回對應的影片數據和標籤
        """
        video_path = self.video_paths[idx]  # 取得影片檔案的完整路徑
        label = self.labels[idx]  # 取得對應的標籤
        frames = torch.load(video_path, map_location="cpu")  # 從 .pt 文件載入影片數據

        # 確保數據的 dtype 為 float32，以防止數據格式錯誤
        if frames.dtype == torch.float16:
            frames = frames.float()

        # 檢查frame維度是否符合標準格式 (C, T, H, W)
        # 如果是 (T, H, W, C)，則轉換為 (C, T, H, W)
        if frames.ndim == 4 and frames.shape[-1] == 3:
            frames = frames.permute(3, 0, 1, 2)  # 轉換維度順序
        elif frames.ndim != 4 or frames.shape[0] != 3:
            raise ValueError(f"影片 {video_path} 格式錯誤，預期形狀 (C, T, H, W)，但得到 {frames.shape}")

        # 如果模式為 raw，則只應用基本轉換，不做數據增強
        # frame (C, T, H, W)
        if self.mode == "raw":
            frames_list = []
            for i in range(frames.shape[1]):  # 遍歷所有影格
                frame = frames[:, i, :, :].clone()  # 取得第 i 幀 ex:(3,32,244,244)->(3,244,244)
                transformed_frame = self.base_transform(frame)  # 套用基本轉換
                frames_list.append(transformed_frame) #合併成list ex:[(3,244,244),(3,244,244)...(3,244,244)]

        # 如果模式為 augmented
        elif self.mode == "augmented":
            frames_list = []
            for i in range(frames.shape[1]):  # 遍歷所有影格
                frame = frames[:, i, :, :].clone()  # 取得第 i 幀 ex:(3,32,244,244)->(3,244,244)
                frame = F.to_pil_image(frame)  # 轉換為 PIL 圖像
                frame = self.augmentation_transform(frame)  # 套用數據增強
                frame = F.to_tensor(frame)  # 轉換回 Tensor 格式
                frame = self.base_transform(frame)  # 再次應用基本轉換
                frames_list.append(frame) # 合併成list ex:[(3,244,244),(3,244,244)...(3,244,244)]

        # 將frames_list延著第1維度(T幀)堆疊起來 ex:[(3,244,244),(3,244,244)...(3,244,244)]->(3,32,244,244)
        frames_transformed = torch.stack(frames_list, dim=1)

        return frames_transformed, label

