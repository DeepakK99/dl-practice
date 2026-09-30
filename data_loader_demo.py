import torch
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CREATE SOME DATA
# ============================================================

X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000, 1)).float()

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 2. CREATE DATASET
# ============================================================

dataset = TensorDataset(X, y)

print("\nDataset size:")
print(len(dataset))


# ============================================================
# 3. CREATE DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)


# ============================================================
# 4. INSPECT ONE BATCH
# ============================================================

batch_X, batch_y = next(iter(loader))

print("\nOne batch:")
print("batch_X shape:", batch_X.shape)
print("batch_y shape:", batch_y.shape)


# ============================================================
# 5. ITERATE THROUGH THE DATA
# ============================================================

print("\nBatches:")

for batch_number, (batch_X, batch_y) in enumerate(loader):

    if batch_number < 3:
        print(
            f"Batch {batch_number + 1}: "
            f"X={batch_X.shape}, "
            f"y={batch_y.shape}"
        )


print("\nTotal batches:", len(loader))


# ============================================================
# 6. TRY DIFFERENT BATCH SIZES
# ============================================================

for batch_size in [16, 32, 64, 128]:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    print(
        f"batch_size={batch_size:3d} "
        f"→ batches={len(loader)}"
    )