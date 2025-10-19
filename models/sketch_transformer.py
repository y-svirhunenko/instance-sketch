import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Embeddings(nn.Module):
    def __init__(self, device, opt, dropout_p=0.0):
        super(Embeddings, self).__init__()
        self.device = device
        self.hidden_size = opt['hidden_size']
        self.image_size = opt['image_size']
        self.max_strokes = opt['max_stroke']

        self.image_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.location_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

    def forward(self, stroke_images, stroke_locations, num_strokes):
        stroke_images = stroke_images.unsqueeze(2)  # (B, S_max, 1, H, W)
        stroke_locations = stroke_locations.unsqueeze(2)  # (B, S_max, 1, H, W)
        B, S_max, C, H, W = stroke_images.shape

        mask = torch.arange(S_max, device=self.device).expand(B, S_max) < num_strokes.unsqueeze(1)
        x = stroke_images[mask]
        y = stroke_locations[mask]

        x_feat = self.image_branch(x).view(x.size(0), -1)
        y_feat = self.location_branch(y).view(y.size(0), -1)
        fused = torch.cat([x_feat, y_feat], dim=-1)
        emb = self.fusion_mlp(fused)

        embeddings = torch.zeros(B, S_max, self.hidden_size, device=self.device)

        batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(B, S_max)[mask]
        stroke_indices = torch.arange(S_max, device=self.device).unsqueeze(0).expand(B, S_max)[mask]
        embeddings[batch_indices, stroke_indices] = emb

        return embeddings


class SketchSegmentator(nn.Module):
    def __init__(self, device, opt, dropout_p=0.0):
        super(SketchSegmentator, self).__init__()

        self.labels_number = opt['num_labels']
        self.segments_number = opt['num_segments']
        self.hidden_size = opt['hidden_size']
        self.max_stroke = opt['max_stroke']
        self.device = device

        self.embeddings = Embeddings(device=device, opt=opt, dropout_p=dropout_p)

        self.encoder_seg_fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.hidden_size, self.segments_number),
        )

        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.category_embeddings = nn.Embedding(opt['num_labels'], self.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=4,
            dim_feedforward=int(self.hidden_size * 4),
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.max_stroke, self.hidden_size) * 0.02
        )

        self.input_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(dropout_p),
        )

        self.use_sketch_class = False  #

    def forward(self, strokes=None, stroke_locations=None, sketch_stroke_num=None, sketch_category=None):

        B, S_max, C, W = strokes.shape
        stroke_emb = self.embeddings(strokes, stroke_locations, sketch_stroke_num)  # (B, S, D)
        pos_emb = self.position_embeddings.expand(B, -1, -1)

        stroke_plus_pos = torch.cat([stroke_emb, pos_emb], dim=-1)
        stroke_emb_projected = self.input_proj(stroke_plus_pos)  # (B, S, D)

        valid_mask = torch.arange(S_max, device=self.device).expand(B, S_max) < sketch_stroke_num.unsqueeze(1)
        stroke_padding_mask = ~valid_mask
        stroke_emb_projected[stroke_padding_mask] = self.mask_token

        if self.use_sketch_class:
            category_token = self.category_embeddings(sketch_category).unsqueeze(1)  # (B, 1, D)
            embeddings = torch.cat([category_token, stroke_emb_projected], dim=1)  # (B, S+1, D)
            mask = torch.cat([torch.zeros((B, 1), dtype=torch.bool, device=self.device), stroke_padding_mask], dim=1)
        else:
            embeddings = stroke_emb_projected
            mask = stroke_padding_mask

        sequence_output = self.encoder(embeddings, src_key_padding_mask=mask)
        sequence_output[mask] = self.mask_token

        if self.use_sketch_class:
            sequence_output = self.layernorm(sequence_output[:, 1:, :])  # remove category token
        else:
            sequence_output = self.layernorm(sequence_output)

        encoder_seg_outs = self.encoder_seg_fc(sequence_output)

        return encoder_seg_outs
