import os
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_data_loaders
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
    save_dir = os.path.join(base_dir, "experiments", "l2")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader, class_names = get_data_loaders(data_dir, batch_size=32)
    model = StrawberryCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 10

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_test_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc*100:.2f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for i in range(num_epochs):
            f.write(
                f"Epoch {i+1}: "
                f"Train Loss={train_losses[i]:.4f}, "
                f"Train Acc={train_accuracies[i]*100:.2f}%, "
                f"Test Loss={test_losses[i]:.4f}, "
                f"Test Acc={test_accuracies[i]*100:.2f}%\n"
            )

        f.write("\n")
        f.write(f"Final Test Accuracy: {test_accuracies[-1]*100:.2f}%\n")
        f.write(f"Best Test Accuracy: {best_test_acc*100:.2f}%\n")

    print("\nTraining completed.")
    print(f"Final Test Accuracy: {test_accuracies[-1]*100:.2f}%")
    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
