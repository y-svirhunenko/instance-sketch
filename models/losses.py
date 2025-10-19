import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin=1.0, distance_metric='euclidean', reduction='mean'):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction

        if distance_metric not in ['euclidean', 'cosine']:
            raise ValueError("distance_metric must be 'euclidean' or 'cosine'")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    def compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return F.pairwise_distance(x1, x2, p=2)
        else:
            return 1 - F.cosine_similarity(x1, x2)

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor:   (N, D) tensor
            positive: (N, D) tensor
            negative: (N, D) tensor
        Returns:
            loss: scalar if reduction='mean'/'sum', else (N,) tensor
        """

        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        pos_dist = self.compute_distance(anchor, positive)
        neg_dist = self.compute_distance(anchor, negative)

        losses = F.relu(pos_dist - neg_dist + self.margin)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

    def sample_triplets(self, embeddings, leg_ids, sketch_stroke_num):
        """
        Args:
            embeddings: Tensor of shape (B, S_max, H)
            leg_ids: Tensor of shape (B, S_max) with leg labels (-1 for padding)
        Returns:
            anchors, positives, negatives: Each of shape (N, H), where N = total valid triplets
        """
        anchors, positives, negatives = [], [], []
        B, S_max, H = embeddings.shape

        for b in range(B):
            num_strokes = sketch_stroke_num[b]
            valid_embeddings = embeddings[b][:num_strokes]
            valid_leg_ids = leg_ids[b][:num_strokes]

            for i in range(len(valid_embeddings)):

                pos_mask = (valid_leg_ids == valid_leg_ids[i])
                pos_indices = torch.where(pos_mask)[0]
                pos_indices = pos_indices[pos_indices != i]

                neg_indices = torch.where(valid_leg_ids != valid_leg_ids[i])[0]

                if len(pos_indices) == 0 or len(neg_indices) == 0:
                    continue

                positive = valid_embeddings[random.choice(pos_indices)]
                negative = valid_embeddings[random.choice(neg_indices)]

                anchors.append(valid_embeddings[i])
                positives.append(positive)
                negatives.append(negative)

        if len(anchors) == 0:
            return None

        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def get_class_weights(class_freqs, method="sqrt"):
    class_freqs = torch.tensor(class_freqs, dtype=torch.float)
    median_freq = torch.median(class_freqs)
    raw_weights = median_freq / class_freqs

    print(f'Normalization method: {method}')
    if method == "sqrt":
        weights = torch.sqrt(raw_weights)
    elif method == "log":
        weights = torch.log1p(raw_weights)
    else:
        weights = raw_weights

    weights = weights / weights.mean()
    return weights
