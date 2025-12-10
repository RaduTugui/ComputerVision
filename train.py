import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from dataset import ImagesDataset
from augmented_images import TransformedImagesDataset
from architecture import model
from torch.cuda.amp import autocast, GradScaler
import os

# AMP scaler for mixed precision
scaler = GradScaler()

# SETUP DEVICE (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

number_epochs = 40
learning_rate = 0.001
batch_size = 32
image_directory = 'training_data'
validation_split = 0.20

# Load Base Dataset
entire_base_dataset = ImagesDataset(image_dir=image_directory, width=100, height=100)


# We split the indices first to ensure NO overlap between training and validation
base_data_len = len(entire_base_dataset)
training_base_len = int((1 - validation_split) * base_data_len)
validation_base_len = base_data_len - training_base_len

generator = torch.Generator().manual_seed(42)
all_indices = list(range(base_data_len))
train_indices_base_subset, val_indices_subset = random_split(all_indices,
                                                             [training_base_len, validation_base_len],
                                                             generator=generator)

# Extract actual index lists
train_indices_base = train_indices_base_subset.indices
val_indices = val_indices_subset.indices

# Create Validation Set
val_dataset = Subset(entire_base_dataset, val_indices)

#  APPLY AUGMENTATION ONLY TO TRAINING
# Wrap the ENTIRE dataset with augmentation logic
full_augmented_dataset = TransformedImagesDataset(data_set=entire_base_dataset)

# Map the training indices to the augmented dataset's 7x indices
training_indices_augmented = []
for i in train_indices_base:
    for j in range(7):
        training_indices_augmented.append(i * 7 + j)

training_set = Subset(full_augmented_dataset, training_indices_augmented)

# Optimized DataLoaders
# CRITICAL: num_workers=8 allows the CPU to process CLAHE and other preprocessing while GPU trains
NUM_WORKERS = min(8, os.cpu_count() or 1)

training_loader = DataLoader(
    dataset=training_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

validation_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

# Training process
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(number_epochs):
    model.train()
    running_loss = 0.0

    # Unpack 6 items (from TransformedImagesDataset)
    for images, _, _, labels, _, _ in training_loader:
        # CRITICAL: Ensure float32 for GPU
        images = images.to(torch.float32)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels.long())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_training_loss = running_loss / len(training_loader)

    # Validation
    model.eval()
    correct_predicted = 0
    total = 0
    loss_value = 0.0

    # Unpack 4 items (from standard ImagesDataset)
    with torch.no_grad():
        for images, labels, _, _ in validation_loader:
            images = images.to(torch.float32)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels.long())

            loss_value += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct_predicted += (predicted == labels).sum().item()

    avg_loss_value = loss_value / len(validation_loader)
    val_accuracy = correct_predicted / total

    print(
        f'Epoch [{epoch + 1}/{number_epochs}], Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_loss_value:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Save and Package
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")

# Final Eval
model.eval()
correct_predicted_final = 0
total_final = 0
with torch.no_grad():
    for images, labels, _, _ in validation_loader:
        images = images.to(torch.float32).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast():
            outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_final += labels.size(0)
        correct_predicted_final += (predicted == labels).sum().item()

final_accuracy = correct_predicted_final / total_final
print(f'Final Validation Accuracy: {final_accuracy:.4f}')

