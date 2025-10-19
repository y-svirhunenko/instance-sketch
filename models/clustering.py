import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Embeddings(nn.Module):

    def __init__(self, device, opt):
        super(Embeddings, self).__init__()
        self.device = device
        self.hidden_size = opt['hidden_size']
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * opt['image_size'] // 4 * opt['image_size'] // 4, opt['hidden_size'])

    def forward(self, stroke_images, num_strokes):
        stroke_images = stroke_images.unsqueeze(2)
        B, S_max, C, H, W = stroke_images.shape

        x = stroke_images
        mask = torch.arange(S_max, device=self.device).expand(B, S_max) < num_strokes.unsqueeze(1)
        x = x[mask]

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        emb = self.fc1(x)

        embeddings = torch.zeros(B, S_max, self.hidden_size, device=self.device)
        batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(B, S_max)[mask]
        stroke_indices = torch.arange(S_max, device=self.device).unsqueeze(0).expand(B, S_max)[mask]
        embeddings[batch_indices, stroke_indices] = emb

        return embeddings


class Leg_Encoder(nn.Module):

    def __init__(self, device, opt):
        super(Leg_Encoder, self).__init__()

        self.device = device
        self.max_stroke = opt['max_stroke']
        self.hidden_size = opt['hidden_size']
        self.encoder_seg_fc = nn.Linear(self.hidden_size, opt['embedding_size'])

        self.embeddings = Embeddings(device, opt)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.stroke_embeddings = nn.Embedding(5, self.hidden_size, padding_idx=4)  ###

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
        )

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, batch_first=True)
        self.fuse_proj = nn.Linear(3 * self.hidden_size, self.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=4,
            dim_feedforward=int(64 * 4),
            dropout=0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.max_stroke, self.hidden_size) * 0.02
        )

    def forward(self, strokes, sketch_stroke_num, stroke_class=None):
        B, S_max = stroke_class.shape

        embedding_output_transformer = self.embeddings(strokes, sketch_stroke_num)
        stroke_token = self.stroke_embeddings(stroke_class)  # (B, S_max, D)

        valid_mask = torch.arange(S_max, device=self.device).expand(B, S_max) < sketch_stroke_num.unsqueeze(1)
        stroke_padding_mask = ~valid_mask

        position_embeddings = self.position_embeddings.expand(B, -1, -1)

        fused = torch.cat([embedding_output_transformer, stroke_token, position_embeddings], dim=-1)  # (B, S_max, 3*D)
        fused = self.fuse_proj(fused)

        fused[stroke_padding_mask] = self.mask_token
        sequence_output = self.encoder(fused, src_key_padding_mask=stroke_padding_mask)
        sequence_output[stroke_padding_mask] = self.mask_token

        sequence_output = self.layernorm(sequence_output)
        sequence_output = self.encoder_seg_fc(sequence_output)

        return sequence_output
