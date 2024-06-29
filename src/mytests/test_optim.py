import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

model = nn.Sequential(
    nn.Linear(10, 10),
    nn.Linear(10, 20),
)
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

x = torch.randn(2, 10)
for _ in range(10):
    y = model(x).sum()
    optimizer.zero_grad()
    y.backward()
    optimizer.step()
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])

