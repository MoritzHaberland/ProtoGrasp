import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import ToTensor
from data_loader.ocid_loader import OcidDataset
from model import get_model
from modular_configs.configs.dataset_config import get_dataset_config

# Parameters

config = get_dataset_config()
# Setze die erforderlichen Verzeichnisse
split_file_path = config.split_file_path  # Pfad zum Split-Datei
root_path = config.root_path  # Pfad zum Root-Verzeichnis des Datasets
split_name = 'training_0'  # oder 'val', 'test' je nach deinem Split
number_of_angle_classes = 18
batch_size = 4
num_epochs = 10

# Create dataset
dataset = OcidDataset(split_file_path, root_path, split_name, None, number_of_angle_classes)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Define the number of classes (including background)
num_classes = number_of_angle_classes + 1
model = get_model(num_classes)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Zero the gradients, backpropagation, and optimization step
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(i)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item():.4f}")

# Save the model
torch.save(model.state_dict(), "faster_rcnn_model.pth")
