# **VIVIT**

VIVIT 是一個基於 **Video Vision Transformer (ViViT)** 的深度學習模型，使用 **HMDB51** 進行簡單測試。該專案提供 **數據處理、模型訓練、評估和推理**，並支持四種 ViViT 模型。

---

## **目錄**

- [簡介](#簡介)
- [檔案說明](#檔案說明)
- [模型介紹](#模型介紹)
- [安裝](#安裝)
- [數據處理](#數據處理)
- [模型訓練](#模型訓練)
- [模型架構](#模型架構)
- [模型評估](#模型評估)
- [影片預測](#影片預測)
- [實驗結果](#實驗結果)
- [其他檔案說明](#其他檔案說明)
- [參考](#參考)

---

## **簡介**

ViViT (Video Vision Transformer) 是一種專為影片分類設計的 **Transformer** 模型。本專案使用 **HMDB51** 進行簡單測試，並實現以下功能：

- 影片預處理與增強
- ViViT 模型訓練與測試
- 影片動作分類推理
- 支援多種 ViViT 變體

---
## **檔案說明**

| 檔案名稱               | 描述                           |
|----------------------|--------------------------------|
| **`model.py`**       | 定義 4 種 ViViT 模型架構         |
| **`model_util.py`**  | `model.py` 的輔助函式，包括 Transformer Encoder、Attention 機制等 |
| **`preprocess.py`**  | 影片預處理，將原始 HMDB51 影片轉換為 `.pt` 格式 |
| **`dataset.py`**     | PyTorch `Dataset`，處理 HMDB51 的原始數據與增強數據 |
| **`train.py`**       | 訓練 ViViT 模型，支援 AMP、梯度累積、學習率調整等 |
| **`evaluation.py`**  | 使用測試集評估模型表現，計算 Loss 和 Accuracy |
| **`predict.py`**     | 輸入影片，讓模型預測動作類別（支援 Top-K 結果） |
| **`class_to_idx.json`** | 類別名稱與索引的對應表，確保標籤一致 |
| **`requirements.txt`** | 記錄專案所需的 Python 套件清單 |

---
## **模型介紹**

✅ **支援多種 ViViT 變體**

### **模型 1: Spatio-temporal Attention**

此模型對所有 **spatio-temporal tokens** 進行 **Transformer Encoder** 處理，計算所有 token 之間的全局關係。該方法在保留完整時序資訊的同時，能夠最大化捕捉影片內的關鍵動作特徵。然而，由於每個 transformer 層都需處理所有時間與空間資訊，計算成本較高，尤其當輸入影片幀數增加時，計算資源消耗將顯著提高。

### **模型 2: Factorised Encoder**

該模型採用 **兩個 Transformer Encoder**，分別處理 **空間維度** 和 **時間維度**，這類似於 **late fusion** 方法。首先，空間編碼器獨立處理每一幀的特徵，然後時間編碼器對所有幀的輸出進行時序建模。這樣的架構有效降低了計算複雜度，並提升模型對於長時間影片的適應能力。與 Spatio-temporal Attention 模型相比，該方法能夠在保留時序資訊的同時減少運算成本。

### **模型 3: Factorised Self-Attention**

此模型將 **Multi-Head Self-Attention** 進一步拆分為 **空間注意力** 和 **時間注意力** 兩部分，先計算空間注意力，再計算時間注意力。這樣的分解方式可以大幅減少 transformer 的計算需求，使其在計算資源有限的環境下仍能發揮良好效能。該方法在某些影片分類任務中表現良好，特別適用於需要較低計算成本但仍需維持較高準確度的應用場景。

### **模型 4: Factorised Dot-Product Attention**

該模型在計算注意力機制時進行了 **進一步的優化**，通過將 **Dot-Product Attention** 分解為獨立的 **空間注意力** 和 **時間注意力** 兩部分，分別計算後再合併，進一步提高計算效率。該方法在保持與 Spatio-temporal Attention 相似性能的同時，大幅減少了計算資源的使用，使其更適用於資源受限的設備。

✅ **高效的數據預處理**

- 均勻採樣影片幀
- 圖像標準化與增強
- 訓練/測試數據分割

✅ **模型訓練**

- 自動混合精度訓練 (**AMP**)
- 梯度累積與裁剪
- Cosine Annealing 學習率調整

✅ **測試與推理**

- 計算 **Top-1 準確率**
- 對影片進行動作分類

---

## **安裝**

請先確保安裝 **Python 3.8+** 及 PyTorch，然後執行以下指令安裝相依套件：

```bash
pip install -r requirements.txt
```

如果需要使用 **CUDA** 加速，請確保已安裝 **NVIDIA CUDA** 驅動。

## **數據處理**

### **數據集**
本專案使用 **HMDB51** 數據集進行測試，該數據集包含 **51** 個不同的動作類別，每個類別包含多個短影片。

### **下載 HMDB51 數據集**

http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

使用上述連結即可下載壓縮檔，接著解壓縮後就有影片

### **數據預處理**

1. 確保 **HMDB51** 數據集已經下載，並存放於 `hmdb51/` 目錄。
2. 運行 `preprocess.py` 來轉換 `.avi` 影片為 PyTorch Tensor，並存儲至 `preprocessed_data_pt/`。

```bash
python preprocess.py
```

該步驟將：
- 轉換影片為固定幀數 (32 幀)
- 標準化影像尺寸 (224x224)
- 產生 **train/test** 數據集
- 建立 **class_to_idx.json**

---

## **模型訓練**

使用 `train.py` 來訓練 ViViT 模型，由於使用筆電訓練就選擇計算量小的 **ViViT Factorized Dot-Product Attention** 模型。

### **指令說明**

```bash
python train.py --epochs 50 --batch_size 8 --lr 0.001 --weight_decay 0.03 \
                 --train_data preprocessed_data_pt/train --test_data preprocessed_data_pt/test \
                 --save_dir checkpoints --model_name vivit_factorized_dotproduct
```

- `--epochs`：訓練的回合數，決定模型的訓練次數。
- `--batch_size`：批次大小，每次訓練所用的樣本數量。
- `--lr`：學習率，控制模型更新權重的步長。
- `--weight_decay`：權重衰減，防止過擬合的正則化技術。
- `--train_data`：訓練數據集的路徑。
- `--test_data`：測試數據集的路徑。
- `--save_dir`：儲存模型檔案的目錄。
- `--model_name`：選擇要訓練的模型類型。

### **模型參數**
```
in_channels=3, embed_dim=96, patch_size=16, tubelet_size=2,
num_heads=8, mlp_dim=96*3, num_layers=6, num_classes=51,
num_frames=32, img_size=224
```
模型參數

in_channels=3：輸入影像的通道數 (RGB 影像通常為 3 通道)。

embed_dim=96：Transformer 的嵌入維度。

patch_size=16：每個影像空間維度 patch 的大小。

tubelet_size=2：影片時間維度上的片段大小。

num_heads=8：自注意力機制的頭數。

mlp_dim=96*3：MLP 層的隱藏維度。

num_layers=6：Transformer 層數。

num_classes=51：類別數量 (對應 HMDB51 數據集的 51 個動作類別)。

num_frames=32：每個影片的時間幀數。

img_size=224：影像的解析度大小 (224x224)。

---

## **模型評估**

使用 `evaluation.py` 來評估模型的表現。

### **指令說明**

```bash
python evaluation.py --data_path preprocessed_data_pt/test --model_path checkpoints/vivit_epoch_50.pth --model vivit_factorized_dotproduct
```

- `--data_path`：測試數據集的路徑。
- `--model_path`：訓練後的模型權重檔案。
- `--model`：使用的模型類型。

此步驟會計算：
- **測試集 Loss**
- **測試集 Accuracy** (Top-1 準確率)

---

## **影片預測**

使用 `predict.py` 來對影片進行動作分類推理。

### **指令說明**

```bash
python predict.py example.avi checkpoints/vivit_epoch_50.pth --model_name vivit_factorized_dotproduct
```

- `example.avi`：要進行分類的影片檔案。
- `checkpoints/vivit_epoch_50.pth`：訓練好的模型權重。
- `--model_name`：使用的模型類型。

如果需要測試多個影片，可以使用 `test_model.py`：

```bash
python test_model.py --model_path checkpoints/vivit_epoch_50.pth --video_path example.avi --class_json class_to_idx.json
```

- `--model_path`：模型權重檔案的路徑。
- `--video_path`：測試影片的路徑。
- `--class_json`：類別索引對應的 JSON 檔案。

---

## **實驗結果**

```
訓練集準確率
🔥 Epoch 1: Avg Loss 4.0004, Accuracy 0.84%, Time 294.23 sec
🔥 Epoch 5: Avg Loss 3.5283, Accuracy 13.11%, Time 280.34 sec ✅ 測試集: Loss 3.5354, Accuracy 13.08%
🔥 Epoch 10: Avg Loss 3.0687, Accuracy 24.88%, Time 271.38 sec ✅ 測試集: Loss 3.1209, Accuracy 23.40%
🔥 Epoch 25: Avg Loss 2.1565, Accuracy 53.23%, Time 284.60 sec ✅ 測試集: Loss 2.7570, Accuracy 38.44%
🔥 Epoch 35: Avg Loss 1.8215, Accuracy 65.85%, Time 303.14 sec ✅ 測試集: Loss 2.7246, Accuracy 42.59%
🔥 Epoch 50: Avg Loss 1.6307, Accuracy 73.79%, Time 280.25 sec ✅ 測試集: Loss 2.7623, Accuracy 42.88%
```
原始論文提到小數據集的過擬和情況嚴重，hmdb51的結果也試過擬和嚴重，最後訓練50次的測試集實驗結果僅為42.88%，因為僅作模型測試就不做模型改善來提高準確率

---
## **其他檔案說明**

- trained model/vivit_epoch_50.pth :　提供訓練50次的model4模型做測試
- checkpoints : 模型記錄點，每5次訓練後會存模型
- hmdb51資料夾 : 用來存放解壓縮後的hmdb51影片
- preprocess_data_pt : hmdb51影片的pt檔案並分為train和test

---

---
## **參考**

- **ViViT 論文**: [Video Vision Transformer (ViViT)](https://arxiv.org/abs/2103.15691)
- **HMDB51 數據集**: [HMDB51 動作分類](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)

---
