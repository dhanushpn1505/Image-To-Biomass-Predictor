import torch
import torch.nn as nn
import pandas as pd
import os
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load test CSV
test_df = pd.read_csv("test.csv")

class TestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.loc[idx, "sample_id"])

        base_id = sample_id.split("__")[0]
        img_name = base_id + ".jpg"
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, sample_id

# Same transform as training (but no augmentation!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = TestDataset(test_df, "test", transform)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load trained model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("biomass_model.pth"))
model = model.to(device)
model.eval()

predictions = []

with torch.no_grad():
    for images, sample_ids in test_loader:
        images = images.to(device)
        outputs = model(images)

        for i in range(len(sample_ids)):
            predictions.append([sample_ids[i], outputs[i].item()])

# Create submission
submission = pd.DataFrame(predictions, columns=["sample_id", "target"])
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
