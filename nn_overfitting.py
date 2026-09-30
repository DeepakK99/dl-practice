import copy

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
# 2. CREATE A SMALL CLASSIFICATION DATASET
# ============================================================

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_classes=2,
    random_state=42,
)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
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

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# IMPORTANT:
# Validation and test data must use the scaler
# learned from the training data.
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ============================================================
# 5. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# ============================================================
# 6. DATASETS + DATALOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)


# ============================================================
# 7. DEFINE THE NEURAL NETWORK
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

print("\nModel:")
print(model)


# ============================================================
# 8. LOSS FUNCTION
# ============================================================

loss_fn = nn.BCEWithLogitsLoss()


# ============================================================
# 9. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001,
)


# ============================================================
# 10. VALIDATION FUNCTION
# ============================================================

def evaluate(model, X, y):

    model.eval()

    with torch.no_grad():

        logits = model(X)

        loss = loss_fn(
            logits,
            y,
        )

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).float()

        accuracy = (
            predictions == y
        ).float().mean()

    return loss.item(), accuracy.item()


# ============================================================
# 11. TRAINING LOOP
# ============================================================

epochs = 100

best_val_loss = float("inf")
best_model_state = None

for epoch in range(epochs):

    model.train()

    for batch_X, batch_y in train_loader:

        # ----------------------------------------
        # Forward pass
        # ----------------------------------------

        logits = model(batch_X)

        # ----------------------------------------
        # Calculate loss
        # ----------------------------------------

        loss = loss_fn(
            logits,
            batch_y,
        )

        # ----------------------------------------
        # Clear old gradients
        # ----------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------
        # Backpropagation
        # ----------------------------------------

        loss.backward()

        # ----------------------------------------
        # Update parameters
        # ----------------------------------------

        optimizer.step()


    # ========================================================
    # VALIDATION
    # ========================================================

    train_loss, train_accuracy = evaluate(
        model,
        X_train,
        y_train,
    )

    val_loss, val_accuracy = evaluate(
        model,
        X_val,
        y_val,
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )


# ============================================================
# 12. RESTORE BEST VALIDATION MODEL
# ============================================================

model.load_state_dict(best_model_state)


# ============================================================
# 13. FINAL TEST EVALUATION
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


# ============================================================
# 14. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "mlp_classifier.pt",
)

print("\nModel saved to mlp_classifier.pt")


# ============================================================
# 15. LOAD MODEL
# ============================================================

loaded_model = MLP()

loaded_model.load_state_dict(
    torch.load(
        "mlp_classifier.pt",
        weights_only=True,
    )
)

loaded_model.eval()


# ============================================================
# 16. INFERENCE
# ============================================================

sample = X_test[0].unsqueeze(0)

with torch.no_grad():

    logits = loaded_model(sample)

    probability = torch.sigmoid(logits)

    prediction = (
        probability >= 0.5
    ).int()


print("\nSingle Prediction")
print("-----------------")
print("Probability:", probability.item())
print("Prediction:", prediction.item())
print("Actual:", int(y_test[0].item()))