import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbed2D(nn.Module):

    def __init__(self, in_chans=3, embed_dim=1536, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.proj(x) # [(B*T), D, Hp, Wp]
        Hp, Wp = x.shape[-2:]
        x = rearrange(x, "(b t) d h w -> b t (h w) d", b=B, t=T)
        return x, Hp, Wp # [B, T, N, D]

class DividedSpaceTimeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        hidden_dim = int(dim * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, T, N, D]
        B, T, N, D = x.shape

        # 1. Temporal attention: for each spatial patch location, attend over T
        xt = self.temporal_norm(x)
        xt = rearrange(xt, "b t n d -> (b n) t d")
        yt, _ = self.temporal_attn(xt, xt, xt, need_weights=False)
        yt = rearrange(yt, "(b n) t d -> b t n d", b=B, n=N)
        x = x + yt

        # 2. Spatial attention: for each frame, attend over N patches
        xs = self.spatial_norm(x)
        xs = rearrange(xs, "b t n d -> (b t) n d")
        ys, _ = self.spatial_attn(xs, xs, xs, need_weights=False)
        ys = rearrange(ys, "(b t) n d -> b t n d", b=B, t=T)
        x = x + ys

        # 3. MLP
        x = x + self.mlp(self.mlp_norm(x))
        return x

class SpaceTimeVisionEncoder(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int] = 224,
        patch_size: int= 16,
        in_chans:int = 3,
        embed_dim:int = 1024,
        depth:int = 12,
        num_heads:int = 12,
        mlp_ratio:float = 4.0,
        max_frames:int = 32,
        dropout:float = 0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed2D(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        if isinstance(image_size, (tuple, list)):
            num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        else:
            num_patches = (image_size // patch_size) ** 2

        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_frames, num_patches, embed_dim)
        )

        self.blocks = nn.ModuleList([
            DividedSpaceTimeBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, video):
        """
        video: [B, T, C, H, W]
        returns: [B, N, D] — spatial tokens of the current (last) timestep only
        """
        B, T, C, H, W = video.shape

        x, Hp, Wp = self.patch_embed(video)      # [B, T, N, D]
        N = Hp * Wp

        x = x + self.pos_embed[:, :T, :N, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x[:, -1]                          # [B, N, D]


if __name__ == "__main__":
    """
    Input video: [B, T, C, H, W]
    Patch tokens: [B, T, N, D]

    Temporal attention:
        [B*N, T, D]

    Spatial attention:
        [B*T, N, D]
    """

    encoder = SpaceTimeVisionEncoder(
        image_size=(192, 320),
        patch_size=16,
        embed_dim=1024,
        depth=12,
        num_heads=16,
        max_frames=16,
    )

    video = torch.randn(2, 16, 3, 192, 320)
    out = encoder(video)

    print(out.shape)  # [2, 240, 1024]
