import torch
import torch.nn as nn

class TubeletEmbedding(nn.Module):
    # RGB、小立方體經過特徵提取後的壓縮表示、立方體長寬、立方體的時間
    def __init__(self, in_channels, embed_dim, patch_size, tubelet_size, kernel_init_method=None):
        super().__init__()

        # 3D 卷積，將影片切割成小塊，出來的結果是長、寬、時間的塊數
        self.conv3d = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            bias=True
        )

        # 選擇初始化函數
        if kernel_init_method == 'central_frame_initializer':
            self.kernel_initializer = central_frame_initializer()
        elif kernel_init_method == 'average_frame_initializer':
            self.kernel_initializer = average_frame_initializer()
        else:
            self.kernel_initializer = None

        # 初始化權重
        if self.kernel_initializer is not None:
            with torch.no_grad():
                self.conv3d.weight.data = self.kernel_initializer(self.conv3d.weight)

    def forward(self, x):
        # Input shape: (B, C, T, H, W)
        x = self.conv3d(x)  # (B, embed_dim, num_tubelets, num_patches_h, num_patches_w)

        # 取得新形狀
        B, embed_dim, num_tubelets, num_patches_h, num_patches_w = x.shape

        # 重新排列維度以適應 Transformer (B, num_tubelets, num_patches_h, num_patches_w, embed_dim)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # 展平成 (B, N, embed_dim)，其中 N = num_tubelets * num_patches_h * num_patches_w
        num_tokens = num_tubelets * num_patches_h * num_patches_w
        x = x.view(B, num_tokens, embed_dim)

        return x

def central_frame_initializer():
    def init(weight):
        weight.zero_()
        center_time_idx = weight.shape[2] // 2
        weight[:, :, center_time_idx, :, :] = torch.randn_like(weight[:, :, center_time_idx, :, :]) * 0.5
        return weight
    return init

def average_frame_initializer():
    def init(weight):
        avg_weight = weight.mean(dim=2, keepdim=True)
        weight.copy_(avg_weight.expand_as(weight) + torch.randn_like(weight) * 0.01)
        return weight
    return init

#處理時間、空間、class的token embeding
class TokenProcessor:
    """
    封裝 CLS Token、Positional Encoding、Temporal Encoding 的工具類別
    """

    @staticmethod
    def add_cls_token(x, cls_token):
        """加入 CLS Token"""
        B, N, C = x.shape
        cls_token_expanded = cls_token.expand(B, -1, -1)
        return torch.cat((cls_token_expanded, x), dim=1)  # (B, N+1, embed_dim)

    @staticmethod
    def add_positional_embedding(x, pos_embedding):
        """加入 Positional Embedding"""
        return x + pos_embedding[:, :x.shape[1], :]

    @staticmethod
    def temporal_embedding(x, method="cls"):
        """
        method 1
            a video-> n frames, n frames have n cls token，
            so we use n cls token as temperal information

        method 2
            論文 : a global average pooling from the tokens output by the spatial encoder
            a video-> n frames, 每個frame有P個patch，將這p個patch排除cls token後取平均變成1個token，稱作avg_token
            n frames就會有n個avg_token，就可以當成temperal information
        """

        if method == "cls":
            # **使用 CLS Token 作為 Temporal Information**
            return x[:, 0:1]  # 取出 CLS Token (B, 1, embed_dim)

        elif method == "gap":
            # **使用 GAP 壓縮所有 Patch Token，得到該幀的全局資訊**
            return x[:, 1:].mean(dim=1, keepdim=True)  # (B, 1, embed_dim)

        else:
            raise ValueError("`method` 必須是 'cls' 或 'gap'")

#注意力機制
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        # QKV 變換
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 最終輸出層
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Q, K, V 計算並 reshape
        Q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)  # 避免過擬合

        # 加權和
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, C)

        # Dropout 和輸出層
        return self.out(self.dropout(out))

# model1和model2的TransformerEncoderBlock
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, droplayer_p=0.1):
        super().__init__()

        # Droplayer 機率
        self.droplayer_p = droplayer_p

        # 多頭自注意力層
        self.attn = Attention(embed_dim, num_heads, dropout)

        # 用於注意力層前後的 LayerNorm
        self.norm_attn = nn.LayerNorm(embed_dim)

        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        # 用於 mlp 區塊前後的 LayerNorm，命名為 norm_mlp
        self.norm_mlp = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Self-Attention + Residual Connection
        attn_out = self.attn(self.norm_attn(x))
        if self.training and torch.rand(1).item() < self.droplayer_p:
            x = x  # ✅ 隨機跳過該層
        else:
            x = x + self.dropout(attn_out)
        x = self.norm_attn(x)

        # MLP (Feedforward Network) + Residual Connection
        mlp_out = self.mlp(self.norm_mlp(x))
        if self.training and torch.rand(1).item() < self.droplayer_p:
            x = x  # ✅ 隨機跳過該層
        else:
            x = x + self.dropout(mlp_out)
        x = self.norm_mlp(x)

        return x

# transformer layer : 將多個 TransformerEncoderBlock 堆疊起來
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()  # 創建一個空的 ModuleList

        # 使用 for 迴圈來添加 num_layers 個 TransformerEncoderBlock
        for _ in range(num_layers):
            self.layers.append(TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout))

    def forward(self, x):
        #layer : 當前的所在的block， layers : 全部的block
        for layer in self.layers:
            x = layer(x)
        return x

# model3的transformer block
class FactorizedTransformerBlock(nn.Module):
    """
    Model 3 (Factorized Self-Attention) 中單一 Block：
      - 先做空間注意力 (Spatial)
      - 再做時間注意力 (Temporal)
      - 最後 MLP
    並有對應的 LayerNorm 與殘差連接
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, droplayer_p=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.droplayer_p = droplayer_p

        # 三個 LayerNorm，對應空間、時間、MLP
        self.norm_spatial = nn.LayerNorm(embed_dim)
        self.norm_temporal = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

        # Attention
        self.attn_spatial = Attention(embed_dim, num_heads, dropout)
        self.attn_temporal = Attention(embed_dim, num_heads, dropout)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    #Stochastic Depth
    def get_drop_pattern(self, x, deterministic):
        if not deterministic and self.droplayer_p:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            return torch.bernoulli(torch.full(shape, self.droplayer_p, device=x.device))
        else:
            return 0.0

    def forward(self, x, T):
        """
        x: (B, N, C)，其中 N = T * P
        B：代表 batch 大小，即一次送入模型的影片數量
        N : 影片的總token數
        T: 總共有多少幀
        P: 每幀照片的packet token數
        C：每個 token 的特徵維度

        把影片的token數(N)拆成每幀照片的packet token數(P)以及T幀照片(T)
        空間注意力 : 計算同一幀中各個 patch 之間的關係，所以只針對每幀照片裡的所有packet數做注意力機制
                   (B, T, P, C) => (B*T, P, C)，B*T表示所有影片中的所有幀，P為input token
        時間注意力 : 計算同一個 patch 在不同時間點的關係，所以只針對T幀照片做注意力機制
                   (B, T, P, C) => (B*P, T, C)，B*P表示所有影片的每一幀的packet token數，T為input token
        """

        B, N, C = x.shape
        P = N // T  # 每幀的 patch 數

        # (1) 空間注意力 (Spatial)
        # 先 reshape => (B, T, P, C)
        x_spatial = x.view(B, T, P, C)
        # 再展開到 (B*T, P, C) 做注意力
        x_spatial_reshaped = x_spatial.view(B * T, P, C)
        x_spatial_normed = self.norm_spatial(x_spatial_reshaped)
        attn_spatial = self.attn_spatial(x_spatial_normed)  # (B*T, P, C)

        # 加入 Stochastic Depth
        drop_pattern = self.get_drop_pattern(x_spatial_reshaped, self.training)
        x_spatial_reshaped = x_spatial_reshaped * (1.0 - drop_pattern) + self.dropout(attn_spatial)
        # reshape 回 (B, T, P, C)
        x_spatial = x_spatial_reshaped.view(B, T, P, C)

        # (2) 時間注意力 (Temporal)
        # 先換軸 => (B, P, T, C)，把時間軸跟 patch 軸對調
        x_temporal = x_spatial.permute(0, 2, 1, 3).contiguous()  # (B, P, T, C)
        # reshape => (B*P, T, C)
        x_temporal_reshaped = x_temporal.view(B * P, T, C)
        x_temporal_normed = self.norm_temporal(x_temporal_reshaped)
        attn_temporal = self.attn_temporal(x_temporal_normed)  # (B*P, T, C)

        # 加入 Stochastic Depth
        drop_pattern = self.get_drop_pattern(x_temporal_reshaped, self.training)
        x_temporal_reshaped = x_temporal_reshaped * (1.0 - drop_pattern) + self.dropout(attn_temporal)
        # reshape 回 (B, P, T, C)，再 permute 回 (B, T, P, C)
        x_temporal = x_temporal_reshaped.view(B, P, T, C).permute(0, 2, 1, 3).contiguous()

        # (3) MLP
        x_out = x_temporal.view(B, N, C)  # 重新攤平 => (B, N, C)
        x_mlp_normed = self.norm_mlp(x_out)
        mlp_out = self.mlp(x_mlp_normed)  # (B, N, C)

        # 加入 Stochastic Depth
        drop_pattern = self.get_drop_pattern(x_temporal_reshaped, self.training)
        x_temporal_reshaped = x_temporal_reshaped * (1.0 - drop_pattern) + self.dropout(attn_temporal)

        return x_out

# 堆疊多層 FactorizedTransformerBlock
class FactorizedTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, num_layers, dropout=0.1, droplayer_p=0.0):
        super().__init__()
        self.layers = nn.ModuleList()  # 創建一個空的 ModuleList

        # 使用 for 迴圈來添加 num_layers 個 TransformerEncoderBlock
        for _ in range(num_layers):
            self.layers.append(FactorizedTransformerBlock(embed_dim, num_heads, mlp_dim, dropout, droplayer_p))

    def forward(self, x, T):
        """
        x: (B, N, C)
        T: 幀數
        """
        for layer in self.layers:
            x = layer(x, T)
        return x

# model4的注意力機制 dot-product attention
class FactorizedDotProductAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 注意力頭數必須是偶數，這樣才能平分成兩組
        assert num_heads % 2 == 0, "Number of heads must be even for factorized attention."
        self.head_dim = embed_dim // num_heads  # 單一個頭的維度

        # 線性投影：產生 Query, Key, Value
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, T):
        """
        x: (B, N, C)，其中 N = T * P (P: 每幀的 patch 數)
        T: 有效的幀數（例如原始幀數除以 tubelet_size）
        """
        B, N, C = x.shape
        P = N // T  # 每幀的 patch 數

        # 線性變換得到 Q, K, V，形狀皆為 (B, N, C)
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # 分組頭：前 half 用於空間注意力，後 half 用於時間注意力
        num_heads_half = self.num_heads // 2

        # -------- 空間注意力 --------
        # 先將 Q, K, V 重塑成 (B, T, P, num_heads, head_dim)
        Q = Q.view(B, T, P, self.num_heads, self.head_dim)
        K = K.view(B, T, P, self.num_heads, self.head_dim)
        V = V.view(B, T, P, self.num_heads, self.head_dim)

        # 取出前半數頭 (用於空間注意力)，形狀：(B, T, P, num_heads_half, head_dim)
        Q_space = Q[:, :, :, :num_heads_half, :]
        K_space = K[:, :, :, :num_heads_half, :]
        V_space = V[:, :, :, :num_heads_half, :]
        # 為了讓注意力在 patch 維度上計算，我們將 head 與 patch 維度交換：
        Q_space = Q_space.permute(0, 1, 3, 2, 4)  # (B, T, num_heads_half, P, head_dim)
        K_space = K_space.permute(0, 1, 3, 2, 4)  # (B, T, num_heads_half, P, head_dim)
        V_space = V_space.permute(0, 1, 3, 2, 4)  # (B, T, num_heads_half, P, head_dim)
        # 計算空間注意力分數
        attn_scores_space = torch.matmul(Q_space, K_space.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 對注意力分數在最後一個維度（patch 維度）做 softmax
        attn_weights_space = torch.softmax(attn_scores_space, dim=-1)
        attn_weights_space = self.dropout(attn_weights_space)
        # 計算空間注意力輸出
        out_space = torch.matmul(attn_weights_space, V_space)  # (B, T, num_heads_half, P, head_dim)
        # 還原順序：交換回 patch 與 head 維度
        out_space = out_space.permute(0, 1, 3, 2, 4).contiguous()  # (B, T, P, num_heads_half, head_dim)

        # -------- 時間注意力 --------
        # 取出後半數頭 (用於時間注意力)，形狀：(B, T, P, num_heads_half, head_dim)
        Q_time = Q[:, :, :, num_heads_half:, :]
        K_time = K[:, :, :, num_heads_half:, :]
        V_time = V[:, :, :, num_heads_half:, :]
        # 將時間與 patch 維度交換，使得每個 patch 成為一個序列
        Q_time = Q_time.permute(0, 2, 3, 1, 4)  # (B, P, num_heads_half, T, head_dim)
        K_time = K_time.permute(0, 2, 3, 1, 4)  # (B, P, num_heads_half, T, head_dim)
        V_time = V_time.permute(0, 2, 3, 1, 4)  # (B, P, num_heads_half, T, head_dim)
        # 計算時間注意力分數
        attn_scores_time = torch.matmul(Q_time, K_time.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 對注意力分數在最後一個維度（時間維度）做 softmax
        attn_weights_time = torch.softmax(attn_scores_time, dim=-1)
        attn_weights_time = self.dropout(attn_weights_time)
        # 計算時間注意力輸出
        out_time = torch.matmul(attn_weights_time, V_time)  # (B, P, num_heads_half, T, head_dim)
        # 還原順序：調整回 (B, T, P, num_heads_half, head_dim)
        out_time = out_time.permute(0, 3, 1, 2, 4).contiguous()  # (B, T, P, num_heads_half, head_dim)

        # -------- 合併兩組注意力結果 --------
        # 沿著頭數維度（第 4 維度）將空間與時間注意力的輸出合併
        out = torch.cat([out_space, out_time], dim=3)  # (B, T, P, num_heads, head_dim)
        # 攤平成 (B, T*P, embed_dim)
        out = out.view(B, T * P, self.embed_dim)
        out = self.out_linear(out)
        out = self.dropout(out)
        return out

# model4的transformer block
class FactorizedDotProductAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.attn = FactorizedDotProductAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, T):
        # Attention + 殘差連接
        x_norm = self.norm_attn(x)
        attn_out = self.attn(x_norm, T)
        x = x + self.dropout(attn_out)
        # MLP + 殘差連接
        x_norm = self.norm_mlp(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)
        return x

# 堆疊多層 FactorizedDotProductAttentionBlock
class FactorizedDotProductAttentionEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()  # 創建一個空的 ModuleList

        # 使用 for 迴圈來添加 num_layers 個 TransformerEncoderBlock
        for _ in range(num_layers):
            self.layers.append(FactorizedDotProductAttentionBlock(embed_dim, num_heads, mlp_dim, dropout))

    def forward(self, x, T):
        for layer in self.layers:
            x = layer(x, T)
        return x
