import torch
import torch.nn as nn
from model_util import (TubeletEmbedding, Attention, TransformerEncoder, TokenProcessor
    , FactorizedTransformerEncoder, FactorizedDotProductAttentionEncoder)

#MODEL 1
class model1(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, tubelet_size,
                 num_heads, mlp_dim, num_layers, num_classes, num_frames, img_size):
        super().__init__()

        # Tubelet Embedding（影片 Token 化）
        self.tubelet_embedding = TubeletEmbedding(in_channels, embed_dim, patch_size, tubelet_size)

        # 動態計算 Token 數量
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.num_tokens = num_frames * num_patches

        # CLS Token + 位置編碼
        # (1, 1, embed_dim) (batch， token數， Transformer Token 的維度)
        # nn.Parameter變成可訓練參數
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # (1, 影片的token數+cls(1 token), Transformer Token 的維度)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens + 1, embed_dim))

        # Transformer Encoder
        self.transformer = TransformerEncoder(embed_dim, num_heads, mlp_dim, num_layers)

        # LayerNorm + MLP Head（輸出分類結果）
        self.norm = nn.LayerNorm(embed_dim)
        #將原embed_dim維度投射成分類個數
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward Pass 流程：
        1️⃣ 影片進入 Tubelet Embedding（轉換成 tokens）
        2️⃣ 加入 CLS Token
        3️⃣ 加入位置編碼（時間 + 空間資訊）
        4️⃣ 進入 Transformer Encoder
        5️⃣ 取 CLS Token 並送入 MLP Head 進行分類
        """
        B, C, T, H, W = x.shape  # B: 批次大小, C:通道數, T:幀數, H:高, W:寬

        # Step 1: 影片 Token 化
        x = self.tubelet_embedding(x)  # (B, N, embed_dim)
        B, N, C = x.shape  # 重新取出 B, N, C

        # Step 2: 加入 CLS Token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim) -1表示不改變
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, embed_dim)

        # Step 3: 加入位置編碼
        #[:, :N+1, :] 第一個:和第三個:代表取整個 batch 維度和所有的 embed_dim、
        # :N+1 代表取 剛好跟輸入 Token 數（N+1 個）對應的部分
        x += self.pos_embedding[:, :N+1, :]

        # Step 4: Transformer Encoder
        x = self.transformer(x)

        # Step 5: 取 CLS Token 並送入 MLP Head 進行分類
        # x[:, 0]表示所有batch和第0個token
        x = self.norm(x[:, 0])  # 取出 CLS Token並正規化 (B, embed_dim)
        x = self.mlp_head(x)  # 輸出分類結果 (B, num_classes)

        return x

#MODEL 2
class ViViT_Factorized(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, tubelet_size, num_heads, mlp_dim,
                 num_layers_spatial, num_layers_temporal, num_classes, num_frames, img_size, droplayer_p):
        super().__init__()

        # **Tubelet Embedding**
        self.tubelet_embedding = TubeletEmbedding(in_channels, embed_dim, patch_size, tubelet_size)

        # **計算影片的 Token 數量**
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        effective_num_frames = num_frames // tubelet_size  # 下採樣後的幀數
        self.num_tokens = effective_num_frames * num_patches  # 正確的 token 數量

        # **CLS Token & 位置編碼**
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens + 1, embed_dim))

        # **空間 Transformer（Spatial Transformer Encoder）**
        self.spatial_transformer = TransformerEncoder(embed_dim, num_heads, mlp_dim, num_layers_spatial, droplayer_p)

        # **時間 Transformer（Temporal Transformer Encoder），層數/2用來減緩過擬和**
        self.temporal_transformer = TransformerEncoder(embed_dim, num_heads, mlp_dim, num_layers_temporal//2, droplayer_p)

        # **LayerNorm + MLP Head（分類）**
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape  # B: 批次大小, C:通道數, T:幀數, H:高, W:寬

        # Step 1: 影片 Token 化
        x = self.tubelet_embedding(x)  # (B, N, embed_dim)

        # Step 2: 加入 CLS Token & 位置編碼
        x = TokenProcessor.add_cls_token(x, self.cls_token)  # (B, N+1, embed_dim)
        x = TokenProcessor.add_positional_embedding(x, self.pos_embedding)  # (B, N+1, embed_dim)

        # Step 3: **空間 Transformer**
        x = self.spatial_transformer(x)  # (B, N+1, embed_dim)

        # Step 4: **時間壓縮**
        temporal_method = "gap"
        patch_tokens = TokenProcessor.temporal_embedding(x, method=temporal_method)  # (B, 1, embed_dim)
        x = torch.cat((x[:, 0:1], patch_tokens), dim=1)  # (B, T+1, embed_dim)

        # Step 5: **時間 Transformer**
        x = self.temporal_transformer(x)  # (B, T+1, embed_dim)

        # Step 6: **分類**
        x = self.norm(x[:, 0])  # (B, embed_dim)
        x = self.mlp_head(x)  # (B, num_classes)

        return x

#model3
class ViViT_Factorized_selfAttention(nn.Module):
    """
    Model 3: Factorised Self-Attention 模型
      - 將影片先做 Tubelet Embedding，轉換成 token 序列 (B, N, embed_dim)
      - 加入位置編碼 (不使用 CLS token)
      - 使用 Factorized Transformer Encoder：每一層內先計算空間注意力，再計算時間注意力，
        實現因子化注意力 (見 Eq. (4)-(6) in 原始論文)
      - 最後透過全局平均池化和 MLP head 進行分類
    """

    def __init__(self, in_channels, embed_dim, patch_size, tubelet_size, num_heads, mlp_dim,
                 num_layers, num_classes, num_frames, img_size, dropout=0.3, droplayer_p=0.1):
        super().__init__()

        # Tubelet Embedding：將影片資料轉換成 token 序列
        self.tubelet_embedding = TubeletEmbedding(in_channels, embed_dim, patch_size, tubelet_size)

        # 計算每幀的 patch 數量
        num_patches = (img_size // patch_size) * (img_size // patch_size)

        # 注意：考慮 tubelet_size 對幀數的下採樣
        effective_num_frames = num_frames // tubelet_size  # 例如 8 // 2 = 4
        self.num_tokens = effective_num_frames * num_patches  # 4 * 196 = 784

        # 位置編碼：不使用 CLS token，因此只對 token 序列進行位置編碼
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # 使用 Factorized Transformer Encoder
        self.transformer = FactorizedTransformerEncoder(embed_dim, num_heads, mlp_dim, num_layers, dropout, droplayer_p)

        # 最後用 LayerNorm 與 MLP Head（線性分類層）進行分類
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

        # 紀錄有效的影片幀數，供 Transformer Encoder 內部 reshape 使用
        self.num_frames = effective_num_frames

    def forward(self, x):
        """
        輸入：
          x: (B, C, T, H, W)
              B: 批次大小
              C: 通道數
              T: 幀數 (原始幀數)
              H: 高
              W: 寬
        處理流程：
          1. 使用 Tubelet Embedding 將影片轉換成 token 序列，形狀為 (B, N, embed_dim)，其中 N = (T // tubelet_size) * num_patches
          2. 加入位置編碼 (不含 CLS token)
          3. 使用 Factorized Transformer Encoder，並傳入有效的幀數參數（T // tubelet_size），
             在每一個 Transformer Block 內先計算空間注意力，再計算時間注意力
          4. 對輸出做 LayerNorm，並對 token 維度做全局平均池化 (mean pooling)
          5. 最後經由 MLP Head 進行分類
        """
        B, C, T, H, W = x.shape

        # Step 1: Tubelet Embedding，得到影片 token 序列 (B, N, embed_dim)
        x = self.tubelet_embedding(x)  # (B, N, embed_dim)

        # Step 2: 加入位置編碼 (不含 CLS token)
        x = TokenProcessor.add_positional_embedding(x, self.pos_embedding)  # (B, N, embed_dim)

        # Step 3: Factorized Transformer Encoder 處理，傳入有效的幀數
        x = self.transformer(x, self.num_frames)  # (B, N, embed_dim)

        # Step 4: LayerNorm 與全局平均池化
        x = self.norm(x)  # (B, N, embed_dim)
        x = x.mean(dim=1)  # (B, embed_dim)

        # Step 5: 分類 Head
        x = self.mlp_head(x)  # (B, num_classes)

        return x

class ViViT_Factorized_DotProduct(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, tubelet_size,
                 num_heads, mlp_dim, num_layers, num_classes, num_frames, img_size, dropout=0.1):
        super().__init__()
        self.tubelet_embedding = TubeletEmbedding(in_channels, embed_dim, patch_size, tubelet_size)

        # 計算每幀的 patch 數量
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        effective_num_frames = num_frames // tubelet_size  # 下採樣後的幀數
        self.num_tokens = effective_num_frames * num_patches  # 確保 Token 數量正確

        # 位置編碼（確保與 token 數量匹配）
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # Factorized Dot-Product Attention Encoder
        self.transformer = FactorizedDotProductAttentionEncoder(embed_dim, num_heads, mlp_dim, num_layers, dropout)

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)
        self.num_frames = effective_num_frames

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.tubelet_embedding(x)  # (B, N, embed_dim)

        # 確保 Positional Embedding 與 token 數量一致
        if self.pos_embedding.shape[1] != x.shape[1]:
            self.pos_embedding = nn.Parameter(torch.randn(1, x.shape[1], x.shape[2]).to(x.device))

        x = TokenProcessor.add_positional_embedding(x, self.pos_embedding)
        x = self.transformer(x, self.num_frames)
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, embed_dim)
        x = self.mlp_head(x)  # (B, num_classes)
        return x

