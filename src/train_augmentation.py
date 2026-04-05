import os
import torch
import torch.nn as nn
import torch.optim as optim

from data_loadaug import get_data_loaders
from model import StrawberryCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    save_dir = os.path.join(base_dir, "experiments", "augmentation")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader, class_names = get_data_loaders(data_dir, batch_size=32)
    
    from torch.utils.data import random_split, DataLoader
    full_train_dataset = train_loader.dataset
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    model = StrawberryCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(save_dir, "best_model_earlystop.pth"))
            print(f"  ✓ New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered! Stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nLoaded best model based on validation loss")

    # FINAL TEST - ONLY ONCE AT THE END
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*50}")
    print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print(f"{'='*50}")

    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for i in range(len(train_losses)):
            f.write(
                f"Epoch {i+1}: "
                f"Train Loss={train_losses[i]:.4f}, "
                f"Train Acc={train_accuracies[i]*100:.2f}%, "
                f"Val Loss={val_losses[i]:.4f}, "
                f"Val Acc={val_accuracies[i]*100:.2f}%\n"
            )
        f.write("\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Final Test Accuracy: {test_acc*100:.2f}%\n")

    print("\nTraining completed.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()