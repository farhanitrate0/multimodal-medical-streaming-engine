import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from memmap_stream import MemmapDataset
from model import get_medical_model


def train_medmnist():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 224px, streamed from disk
    train_dataset = MemmapDataset("train", transform=train_transform)
    val_dataset = MemmapDataset("val", transform=val_transform)

    # balanced batches
    labels = train_dataset.labels.squeeze().astype(int)
    class_counts = [len(labels) - sum(labels), sum(labels)]
    class_weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
    sample_weights = [class_weights[label] for label in labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_medical_model(num_classes=1, freeze_base=False).to(device)  # full fine-tune

    criterion = nn.BCEWithLogitsLoss()

    # low lr so imagenet features adapt gently
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    epochs = 15
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, batch_labels in train_loader:
            images = images.to(device)
            batch_labels = batch_labels.float().view(-1).to(device)  # flatten to 1D (defensive)

            optimizer.zero_grad()
            outputs = model(images).view(-1)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, batch_labels in val_loader:
                images = images.to(device)
                batch_labels = batch_labels.float().view(-1).to(device)

                outputs = model(images).view(-1)
                val_loss += criterion(outputs, batch_labels).item()

                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "fine_tuned_medical_resnet.pth")
            print("  checkpoint saved")

    print("Done.")


if __name__ == "__main__":
    train_medmnist()
