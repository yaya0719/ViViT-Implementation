# **VIVIT**

VIVIT 是一個基於 **Video Vision Transformer (ViViT)** 的深度學習模型，專為 **HMDB51** 動作識別數據集設計。該專案提供 **數據處理、模型訓練、評估和推理**，並支持多種 ViViT 變體。

---

## **目錄**
- [簡介](#簡介)
- [特性](#特性)
- [安裝](#安裝)
- [數據處理](#數據處理)
- [模型訓練](#模型訓練)
- [模型評估](#模型評估)
- [影片預測](#影片預測)
- [模型結構](#模型結構)
- [參考](#參考)

---

## **簡介**
ViViT (Video Vision Transformer) 是一種專為影片分類設計的 **Transformer** 模型。本專案針對 **HMDB51** 數據集，實現以下功能：
- 影片預處理與增強
- ViViT 模型訓練與測試
- 影片動作分類推理
- 支援多種 ViViT 變體

---

## **特性**
✅ **支援多種 ViViT 變體**
- `model1`: 標準 Transformer
- `vivit_factorized`: 空間與時間編碼因子化
- `vivit_factorized_selfattention`: 自注意力因子化
- `vivit_factorized_dotproduct`: 點積注意力因子化

✅ **高效的數據預處理**
- 均勻採樣影片幀
- 圖像標準化與增強
- 訓練/測試數據分割

✅ **模型訓練**
- 自動混合精度訓練 (**AMP**)
- 梯度累積與裁剪
- Cosine Annealing 學習率調整

✅ **測試與推理**
- 計算 **Top-5 準確率**
- 對影片進行動作分類

---

## **安裝**
請先確保安裝 **Python 3.8+** 及 PyTorch，然後執行以下指令安裝相依套件：

```bash
pip install -r requirements.txt
```

如果需要使用 **CUDA** 加速，請確保已安裝 **NVIDIA CUDA** 驅動。

---

## **數據處理**
### **1️⃣ 下載 HMDB51 數據集**
請至 [HMDB51 官網](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) 下載 **AVI 影片** 並解壓縮至 `hmdb51/` 資料夾。

### **2️⃣ 預處理影片**
運行 **preprocess.py** 來將 `.avi` 影片轉換為 PyTorch Tensor 並存儲至 `preprocessed_data_pt/`：

```bash
python preprocess.py
```

這將：
- 轉換影片為固定幀數 (32 幀)
- 標準化影像尺寸 (224x224)
- 產生 **train/test** 數據集
- 建立 **class_to_idx.json**

---

## **模型訓練**
使用 `train.py` 來訓練 ViViT 模型，預設使用 **ViViT Factorized Self-Attention**：

```bash
python train.py --epochs 50 --batch_size 8 --lr 3e-4 --model_name vivit_factorized_selfattention
```

📌 **可選參數**
- `--epochs`: 訓練 Epoch 數 (預設: 50)
- `--batch_size`: 批次大小 (預設: 8)
- `--lr`: 初始學習率 (預設: 3e-4)
- `--model_name`: 選擇模型 (`model1`, `vivit_factorized`, `vivit_factorized_selfattention`, `vivit_factorized_dotproduct`)

📝 **訓練過程**
- 使用 **AMP** 進行混合精度訓練
- 10% 訓練數據將應用數據增強
- 每 5 個 Epoch 會保存 `.pth` 模型

---

## **模型評估**
使用 `evaluation.py` 評估訓練後的 ViViT 模型：

```bash
python evaluation.py --data_path preprocessed_data_pt/test --model_path checkpoints/vivit_epoch_50.pth --model vivit_factorized_selfattention
```

📌 **可選參數**
- `--data_path`: 測試數據集路徑
- `--model_path`: 訓練好的模型 (`.pth` 檔案)
- `--model`: 選擇模型類型

---

## **影片預測**
使用 `predict.py` 進行影片分類推理：

```bash
python predict.py <video_path> <model_path> --model_name vivit_factorized_selfattention
```

📌 **範例**
```bash
python predict.py example.avi checkpoints/vivit_epoch_50.pth
```

---

## **模型結構**
| 模型 | 空間 Transformer | 時間 Transformer | 特性 |
|------|----------------|----------------|------|
| **model1** | ✅ | ✅ | 標準 Transformer |
| **vivit_factorized** | ✅ | ✅ | 因子化 Transformer |
| **vivit_factorized_selfattention** | ✅ | ✅ | 自注意力因子化 |
| **vivit_factorized_dotproduct** | ✅ | ✅ | 點積注意力因子化 |

---

## **參考**
- **ViViT 論文**: [Video Vision Transformer (ViViT)](https://arxiv.org/abs/2103.15691)
- **HMDB51 數據集**: [HMDB51 動作分類](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)

---

這份 **README.md** 概述了 **VIVIT** 的使用方法，如果有任何修改需求，請讓我知道！ 🚀

