# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:48:43 2025

@author: astratig
"""
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, *inputs):
        self.inputs = inputs

        # Check that all input tensors have the same length (number of samples)
        self.length = len(inputs[0])
        if not all(len(input_tensor) == self.length for input_tensor in inputs):
            raise ValueError("Input tensors must have the same length.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(input_tensor[idx] for input_tensor in self.inputs)

# Define a custom data loader
def create_data_loader(inputs, batch_size, num_workers=0, shuffle=True):
    dataset = MyDataset(*inputs)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return data_loader