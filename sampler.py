import torch
import numpy as np
import random

from collections import defaultdict
from torch.utils.data import Sampler
from typing import List, Dict


class PKSampler(Sampler[List[str]]):  # 每次选 P 个类别，然后从每个类别中选 K 个样本
    def __init__(self, labels: List[str], P: int, K: int):
        self.P, self.K = P, K
        self.labels = np.array(labels)
        self.label_to_index: Dict[str, List[int]] = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.label_to_index[label].append(i)
        self.labels_set = list(self.label_to_index.keys())
        self.labels_set = [
            c for c in self.labels_set if len(self.label_to_index[c]) >= self.K
        ]
        if len(self.labels_set) < self.P:
            raise ValueError(
                f"P={self.P} is larger than the number of classes with at least K={self.K} samples: {len(self.labels_set)}"
            )

    def __iter__(self):
        while True:
            random.shuffle(self.labels_set)
            for i in range(0, len(self.labels_set), self.P):
                batch_labels = self.labels_set[i : i + self.P]
                if len(batch_labels) < self.P:
                    continue
                batch_indices = []
                for label in batch_labels:
                    batch_indices.extend(
                        random.sample(self.label_to_index[label], self.K)
                    )
                yield batch_indices

    def __len__(self):
        return len(self.labels) // (self.P * self.K)
