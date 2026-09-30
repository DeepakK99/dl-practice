import torch
import torch.nn as nn

class CustomerClassifier(nn.Module):
    def __init__(self):
        super(CustomerClassifier, self).__init__()
        
        # 1.1 Anatomy: Setting up the layers
        # Input: 4 raw features -> Output: 8 hidden features
        self.hidden_layer = nn.Linear(in_features=4, out_features=8)
        
        # Non-Linear Activation Function
        self.relu = nn.ReLU()
        
        # Hidden: 8 features -> Output: 3 final classes
        self.output_layer = nn.Linear(in_features=8, out_features=3)
        
        # Final Activation for multi-class classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 1.2 The Engine: Data flowing through the network
        # Step 1: Linear combination (XW + b) for hidden layer
        x = self.hidden_layer(x)
        
        # Step 2: Apply non-linearity (ReLU)
        x = self.relu(x)
        
        # Step 3: Linear combination for output layer
        x = self.output_layer(x)
        
        # Step 4: Turn raw scores into clean probabilities summing to 1.0
        x = self.softmax(x)
        return x

# Let's run a fake batch of data through it like a production API endpoint
if __name__ == "__main__":
    # Create the model instance
    model = CustomerClassifier()

    # Accessing weights directly from a specific layer
    print("Hidden Layer Weights Shape:", model.hidden_layer.weight.shape)
    print("Hidden Layer Biases Shape:", model.hidden_layer.bias.shape)

    # Printing the actual raw numbers inside the state dictionary
    for layer_name, weights_tensor in model.state_dict().items():
        print(f"\nLayer: {layer_name}")
        print(weights_tensor)
    
    # Simulating a batch of 2 customers, each with 4 features (e.g., age, income, clicks, tenure)
    # PyTorch expects data as float Tensors
    mock_input_data = torch.tensor([[35.0, 75000.0, 12.0, 2.0],
                                    [22.0, 20000.0, 45.0, 0.5]], dtype=torch.float32)
    
    # Run the forward pass
    with torch.no_grad(): # Tells PyTorch we are just predicting, not training
        predictions = model(mock_input_data)
        
    print("Output Probability Distributions:\n", predictions)
