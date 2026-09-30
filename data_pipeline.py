import torch
from torch.utils.data import Dataset, DataLoader

# 1. Define the structural schema of your data
class ProductionDataset(Dataset):
    def __init__(self, features, targets):
        # Ensure our data inputs are converted to proper PyTorch Float tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        # Tells the DataLoader how many rows exist in total
        return len(self.features)

    def __getitem__(self, idx):
        # Tells the DataLoader how to fetch one specific item
        return self.features[idx], self.targets[idx]


# --- Usage in Your Application ---
if __name__ == "__main__":
    # Simulating raw numpy array data coming from a data warehouse or database
    import numpy as np
    raw_features = np.random.randn(100, 4) # 100 customers, 4 metrics each
    raw_targets = np.random.randint(0, 3, size=100) # 3 categories
    
    # Instantiate the Dataset
    my_dataset = ProductionDataset(raw_features, raw_targets)
    
    # Create the DataLoader
    my_dataloader = DataLoader(
        dataset=my_dataset, 
        batch_size=16,       # Process data in batches of 16 rows at a time
        shuffle=True,        # Shuffle every epoch to prevent pattern memorization
        num_workers=2        # Use 2 CPU cores to load data in parallel in the background [3]
    )
    
    print(f"Total batches per epoch: {len(my_dataloader)}") # 100 / 16 = 7 batches
    
    # How it integrates seamlessly into our training loop:
    for batch_features, batch_targets in my_dataloader:
        # These are ready to be pushed to a GPU: batch_features.to('cuda')
        print("Batch feature shape:", batch_features.shape)
        print("Batch target shape:", batch_targets.shape)
        break # Just inspecting the first batch
