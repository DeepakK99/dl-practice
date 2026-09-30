import torch
import torch.nn as nn
import torch.optim as optim
from basic_nn import CustomerClassifier

# 1. Instantiate our model from earlier
model = CustomerClassifier()

# 2. Define the Loss Function (CrossEntropy is standard for multi-class)
criterion = nn.CrossEntropyLoss()

# 3. Define an Optimizer (Stochastic Gradient Descent for now)
# We pass it the model's weights so it knows what to modify, and a learning rate (lr)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Simulating 4 real data points (Inputs) and their correct real-world categories (Targets)
# Target categories must be integers: 0, 1, or 2
dummy_inputs = torch.randn(4, 4) # 4 rows, 4 features each
dummy_targets = torch.tensor([0, 2, 1, 0], dtype=torch.long) 

print("Starting Training Loop...")

# Run the training loop for 50 iterations (epochs)
for epoch in range(50):
    # --- STEP 1: Forward Pass ---
    # Put data through the network to get current predictions
    predictions = model(dummy_inputs)
    
    # --- STEP 2: Calculate the Loss ---
    # See how bad the current random guesses are
    loss = criterion(predictions, dummy_targets)
    
    # --- STEP 3: Clear Past Gradients ---
    # PyTorch accumulates gradients by default; we must wipe them clean before backward pass!
    optimizer.zero_grad()
    
    # --- STEP 4: Backpropagation ---
    # Calculate gradients for every weight automatically
    loss.backward()
    
    # --- STEP 5: Gradient Descent Step ---
    # Tweak the weights by a tiny fraction based on the calculated gradients
    optimizer.step()
    
    # Print progress every 10 steps
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50] | Current Loss: {loss.item():.4f}")



###################################
# adam and dynamic learnign rate  #
###################################
import torch
import torch.nn as nn
import torch.optim as optim

model = CustomerClassifier()
criterion = nn.CrossEntropyLoss()

# 1. Swap SGD out for Adam. Default starting learning rate is usually 0.001 (1e-3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Add a StepLR Scheduler. 
# Every 10 epochs, multiply the current learning rate by a gamma of 0.1 (shrink it by 90%)
scheduler = optim.lr_scheduler.StepRate(optimizer, step_size=10, gamma=0.1) if hasattr(optim.lr_scheduler, 'StepRate') else optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# --- Inside the production loop ---
# (After optimizer.step() runs at the end of an epoch):
# scheduler.step() 
