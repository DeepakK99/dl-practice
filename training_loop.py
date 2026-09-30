import torch
import torch.nn as nn

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

if torch.mps.is_available():
    print("GPU:", torch.backends.mps.get_name())


# ============================================================
# 3. CREATE DATA
# ============================================================

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_classes=2,
    random_state=42,
)


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)


# ============================================================
# 5. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ============================================================
# 6. CONVERT TO TENSORS
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32,
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32,
).view(-1, 1)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32,
)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32,
).view(-1, 1)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32,
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32,
).view(-1, 1)


# ============================================================
# 7. DATASET
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train,
)


# ============================================================
# 8. DATALOADER
# ============================================================

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)


print("\nTraining samples:", len(train_dataset))
print("Batch size:", batch_size)
print("Batches per epoch:", len(train_loader))


# ============================================================
# 9. MODEL
# ============================================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


model = MLP()

# Move model to CPU or GPU
model = model.to(device)


print("\nModel device:")
print(next(model.parameters()).device)


# ============================================================
# 10. LOSS + OPTIMIZER
# ============================================================

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001,
)


# ============================================================
# 11. VALIDATION FUNCTION
# ============================================================

def evaluate(model, X, y):

    model.eval()

    # Move validation data to same device as model
    X = X.to(device)
    y = y.to(device)

    with torch.no_grad():

        logits = model(X)

        loss = loss_fn(
            logits,
            y,
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).float()

        accuracy = (
            predictions == y
        ).float().mean()

    return loss.item(), accuracy.item()


# ============================================================
# 12. TRAINING LOOP
# ============================================================

epochs = 50

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    # --------------------------------------------------------
    # Iterate through mini-batches
    # --------------------------------------------------------

    for batch_X, batch_y in train_loader:

        # Move batch to CPU/GPU
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(batch_X)

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = loss_fn(
            logits,
            batch_y,
        )

        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()

        # Accumulate batch loss
        running_loss += loss.item()


    # ========================================================
    # EPOCH METRICS
    # ========================================================

    average_train_loss = (
        running_loss / len(train_loader)
    )

    val_loss, val_accuracy = evaluate(
        model,
        X_val,
        y_val,
    )


    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )


# ============================================================
# 13. FINAL TEST
# ============================================================

test_loss, test_accuracy = evaluate(
    model,
    X_test,
    y_test,
)

print("\nFinal Test Results")
print("------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")