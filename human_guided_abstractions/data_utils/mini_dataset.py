import torch
import numpy as np


# If you want to construct a mini dataset that stores x examples for each label, use this.
class MiniByLabel(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, factor, num_unique_labels, label_idx):
        self.raw_dataset = raw_dataset
        self.factor = factor
        self.label_idx = label_idx
        self.num_unique_labels = num_unique_labels
        self.cache = {}  # Cache from label to list of examples of that label.
        self.cache_size = self._populate_cache()
        self.sorted_keys = sorted(self.cache.keys())

    def _populate_cache(self):
        num_elts = 0
        while num_elts < self.factor * self.num_unique_labels:
            rand_idx = int(np.random.random() * len(self.raw_dataset))
            row = self.raw_dataset.__getitem__(rand_idx)
            x = row[0]
            labels = row[1:]
            label = labels[self.label_idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            cached_entries = self.cache.get(label)
            if cached_entries is None:
                cached_entries = [row]
                self.cache[label] = cached_entries
                num_elts += 1
            elif len(cached_entries) < self.factor:
                self.cache[label].append(row)
                num_elts += 1
            if num_elts == self.factor * self.num_unique_labels:
                return num_elts
        assert False, "Unable to find all the elements you wanted"

    def __len__(self):
        return self.cache_size

    def __getitem__(self, idx):
        # Select an element from the cache.
        key_idx = idx // self.factor
        key = self.sorted_keys[key_idx]
        entries = self.cache.get(key)
        # Now select from within entries to get the particular one.
        sub_idx = idx % self.factor
        entry = entries[sub_idx]
        x = entry[0]
        label0 = entry[1]
        label1 = entry[2]
        if isinstance(label0, torch.Tensor):
            label0 = label0.item()
        if isinstance(label1, torch.Tensor):
            label1 = label1.item()
        return x, label0, label1, idx
