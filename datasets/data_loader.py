import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class JULEDataset(Dataset):
    """Dataset class for JULE."""
    
    def __init__(self, h5_path, transform=None):
        """
        Args:
            h5_path: Path to the h5 file containing the dataset
            transform: Optional transform to be applied on a sample
        """
        self.transform = transform
        
        # Load data from h5 file
        with h5py.File(h5_path, 'r') as f:
            self.data = torch.from_numpy(f['data'][:]).float()
            self.labels = torch.from_numpy(f['labels'][:]).long()
            
        # Normalize data
        if len(self.data.shape) == 3:
            self.data = self.data.unsqueeze(1)  # Add channel dimension
        self.data = self.data / 255.0  # Normalize to [0,1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

def get_dataloader(dataset_name, batch_size=100, shuffle=True):
    """
    Get data loader for a specific dataset.
    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
    Returns:
        dataloader: PyTorch DataLoader object
    """
    dataset = JULEDataset(f'datasets/{dataset_name}/data4torch.h5')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    return dataloader 