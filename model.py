import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=768, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch_size, 3, image_size, image_size)
        x = self.proj(x)  # (batch_size, emb_dim, grid_size, grid_size)
        x = x.flatten(2)  # (batch_size, emb_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, emb_dim)
        return x

class Attention(nn.Module):
    def __init__(self, emb_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads
        
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn_scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        if not hidden_features:
            hidden_features = in_features
        if not out_features:
            out_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = Attention(emb_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(in_features=emb_dim, hidden_features=int(emb_dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        emb_dim=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_dim=emb_dim,
            image_size=image_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.num_patches, emb_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                emb_dim=emb_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, std=0.02)
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits
