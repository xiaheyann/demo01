import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "npu"


# ---------------------------
# FeedForward
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Convolution Module
# ---------------------------
class ConvModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),

            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),

            nn.Conv1d(dim, dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


# ---------------------------
# Self Attention
# ---------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.attn(x, x, x)
        return out


# ---------------------------
# Conformer Block
# ---------------------------
class ConformerBlock(nn.Module):
    def __init__(self, dim, ff_mult=4):
        super().__init__()

        self.ff1 = FeedForward(dim, dim * ff_mult)
        self.attn = SelfAttention(dim)
        self.conv = ConvModule(dim)
        self.ff2 = FeedForward(dim, dim * ff_mult)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)

        return self.norm(x)


# ---------------------------
# Conformer Model
# ---------------------------
class ConformerMNIST(nn.Module):
    def __init__(self, dim=64, depth=4):
        super().__init__()

        self.patch = nn.Linear(28, dim)

        self.blocks = nn.Sequential(
            *[ConformerBlock(dim) for _ in range(depth)]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(dim, 10)

    def forward(self, x):

        x = x.squeeze(1)

        x = self.patch(x)

        x = self.blocks(x)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        return self.fc(x)


# ---------------------------
# 数据
# ---------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)


# ---------------------------
# 模型
# ---------------------------
model = ConformerMNIST().to(device)

print("Running device:", next(model.parameters()).device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------
# 训练
# ---------------------------
epochs = 5

for epoch in range(epochs):

    model.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in pbar:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())


# ---------------------------
# 测试
# ---------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, pred = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

print("Test Accuracy:", correct / total)
