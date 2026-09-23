import torch
from torch import nn


class VisionTransformer(nn.Module):
    """Compact ViT-Tiny: 12 layers, width 192, 3 heads."""
    def __init__(self, num_classes, image_size=224, patch_size=16, width=192, layers=12, heads=3):
        super().__init__()
        self.patch = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        token_count = (image_size // patch_size) ** 2 + 1
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, token_count, width))
        block = nn.TransformerEncoderLayer(d_model=width, nhead=heads,
            dim_feedforward=width * 4, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls.expand(x.size(0), -1, -1), x), dim=1)
        x = self.encoder(x + self.position)
        return self.head(self.norm(x[:, 0]))
