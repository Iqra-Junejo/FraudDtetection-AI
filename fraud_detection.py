# AUTHOR: Iqra Junejo

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

print("Starting training...")

# 1. Generate synthetic data
np.random.seed(42)
n = 10000
amount = np.random.lognormal(5, 1, n)
time_since_last = np.random.randint(1, 1440, n)
location_change = np.random.exponential(10, n)
is_international = np.random.choice([0, 1], n, p=[0.9, 0.1])

# Simple fraud rule for realistic labels
fraud_prob = (
    (amount > 1000) * 0.4 +
    (time_since_last < 10) * 0.3 +
    (location_change > 80) * 0.25 +
    is_international * 0.2
)
labels = np.random.binomial(1, np.clip(fraud_prob, 0, 0.99) * 0.25)

df = pd.DataFrame({
    'amount': amount,
    'time_since_last': time_since_last,
    'location_change': location_change,
    'is_international': is_international,
    'label': labels
})

print(f"Generated {n} transactions → {labels.sum()} fraud cases")

# 2. Normalize
for col in ['amount', 'time_since_last', 'location_change']:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std

# 3. Prepare tensors
X = torch.tensor(df.drop('label', axis=1).values, dtype=torch.float32)
y = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 4. Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = Net()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))  # handle imbalance
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train (only 20 epochs — super fast)
print("Training started...")
for epoch in range(20):
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/20 completed")

# 6. Save model
torch.save(model.state_dict(), "model.pth")
print("SUCCESS! model.pth has been created")
print("You can now run: python fraud_gui.py")
