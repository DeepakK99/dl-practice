import torch
import torch.nn as nn

class RobustCustomerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer 1
        self.hidden_layer = nn.Linear(4, 16) # Upgraded to 16 features
        
        # 1. Layer Normalization: Keeps data scale under control
        self.layer_norm = nn.LayerNorm(16)
        
        self.relu = nn.ReLU()
        
        # 2. Dropout: Randomly deactivates 25% of neurons during training
        self.dropout = nn.Dropout(p=0.25)
        
        # Output Layer
        self.output_layer = nn.Linear(16, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.layer_norm(x)   # Normalize right after the linear transformation
        x = self.relu(x)
        x = self.dropout(x)     # Apply dropout before sending to output
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = RobustCustomerClassifier()
    mock_data = torch.randn(2, 4)
    
    # --- PRODUCTION PATTERN: Mode Switching ---
    
    # 1. Set model to TRAINING mode (Activates Dropout)
    model.train()
    print("Training mode output (with dropout):\n", model(mock_data))
    
    # 2. Set model to EVALUATION mode (Deactivates Dropout)
    model.eval()
    with torch.no_grad():
        print("\nInference mode output (stable predictions):\n", model(mock_data))
