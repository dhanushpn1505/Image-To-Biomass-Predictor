import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load CSV
df = pd.read_csv("train.csv")

print("Columns:", df.columns)
print("Total samples:", len(df))

# Make sure required columns exist
if "sample_id" not in df.columns or "target" not in df.columns:
    raise ValueError("CSV must contain 'sample_id' and 'target' columns")

class BiomassDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = self.df.loc[idx, "sample_id"]
        label = self.df.loc[idx, "target"]

        # Split at double underscore
        base_id = sample_id.split("__")[0]

        img_name = base_id + ".jpg"
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Train / Validation split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = BiomassDataset(train_df, "train", transform)
val_dataset = BiomassDataset(val_df, "train", transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

# Validation
model.eval()
preds = []
actuals = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)

        outputs = model(images)

        preds.extend(outputs.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

rmse = np.sqrt(mean_squared_error(actuals, preds))
print("Validation RMSE:", rmse)

# Save model
torch.save(model.state_dict(), "biomass_model.pth")
print("Model saved successfully.")