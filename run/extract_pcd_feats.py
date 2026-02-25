import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class VoxelViTEncoder(nn.Module):
    def __init__(self, volume_size=10, patch_size=2, in_channels=1, embed_dim=128, depth=4, heads=4):
        super().__init__()
        # パッチの数: (10 / 2)^3 = 125個
        self.grid_size = volume_size // patch_size
        self.num_patches = self.grid_size ** 3
        # 1パッチあたりのデータ量: 1チャンネル * (2 * 2 * 2) = 8
        self.patch_dim = in_channels * (patch_size ** 3)

        # einopsを使って、3Dボクセルをパッチのシーケンスに平坦化し、次元をembed_dimに投影
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', 
                      p1=patch_size, p2=patch_size, p3=patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 位置エンコーディングと[CLS]トークン
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # PyTorch標準のTransformerEncoderLayerを利用
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True # 最近のViTはPre-Normが主流
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        b = x.shape[0]
        
        # ボクセルをパッチのシーケンスに変換: (B, 125, 128)
        tokens = self.to_patch_embedding(x)

        # [CLS]トークンをバッチサイズ分複製して結合: (B, 126, 128)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, tokens), dim=1)
        
        # 位置エンコーディングを加算
        x += self.pos_embedding

        # Transformerに入力
        x = self.transformer(x)
        
        # 先頭の[CLS]トークンを128次元の潜在表現（Latent Vector）として抽出
        latent = x[:, 0]
        return latent

class VoxelViTDecoder(nn.Module):
    def __init__(self, volume_size=10, patch_size=2, out_channels=1, embed_dim=128):
        super().__init__()
        self.grid_size = volume_size // patch_size
        self.num_patches = self.grid_size ** 3

        # 128次元のベクトルを、125個のパッチ情報に展開
        self.latent_to_patches = nn.Sequential(
            nn.Linear(embed_dim, self.num_patches * embed_dim),
            nn.GELU()
        )

        # 転置畳み込みで空間を拡大 (5x5x5 -> 10x10x10)
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, out_channels, kernel_size=3, padding=1),
            # 論文の入力が0〜0.1でクリップされているため、0〜1に収めるかReLU等で調整
            # ここでは一般的なSigmoidを使用し、学習時にスケールを合わせる想定
            nn.Sigmoid() 
        )

    def forward(self, latent):
        # latent: (B, 128)
        
        # パッチ情報を生成: (B, 125 * 128)
        patches = self.latent_to_patches(latent)
        
        # einopsを使って1次元のシーケンスを3Dグリッドに再構築: (B, 128, 5, 5, 5)
        grid = rearrange(patches, 'b (x y z d) -> b d x y z', 
                         x=self.grid_size, y=self.grid_size, z=self.grid_size, d=128)
        
        # 元のボクセルサイズにアップサンプル: (B, 1, 10, 10, 10)
        reconstructed = self.upsample(grid)
        return reconstructed

class VoxelViTAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = VoxelViTEncoder(**kwargs)
        self.decoder = VoxelViTDecoder(**kwargs)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# ==========================================
# 動作確認
# ==========================================
if __name__ == "__main__":
    # バッチサイズ8、1チャンネル、10x10x10のダミー入力
    dummy_input = torch.rand(8, 1, 10, 10, 10)
    
    # モデルの初期化
    model = VoxelViTAutoEncoder(volume_size=10, patch_size=2, in_channels=1, embed_dim=128)
    
    # 順伝播
    reconstructed, latent = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Latent shape: {latent.shape} -> (Diffusionモデルへ渡す特徴量)")
    print(f"Reconstructed shape: {reconstructed.shape}")