import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):
    # Set directory paths for training and testing sets
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Define image transformations and preprocessing steps [cite: 21, 40]
    # Resize to 128x128 and apply standard ImageNet normalization for better convergence
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets using the ImageFolder structure (labels derived from folder names)
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)

    # Initialize DataLoaders for batch processing [cite: 27, 35]
    # Shuffle enabled for training to improve model generalization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_dataset.classes

# TASK 7: Verification block to ensure the data pipeline is functional 
if __name__ == '__main__':
    # Determine the base data path relative to the script location
    base_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    test_batch_size = 16
    
    print("Starting Data Pipeline Verification...\n")
    
    try:
        train_loader, test_loader, classes = get_data_loaders(base_data_path, batch_size=test_batch_size)
        print(f"Discovered Classes: {classes}")
        print(f"Total Training Batches: {len(train_loader)}")
        print(f"Total Testing Batches: {len(test_loader)}\n")
        
        # Attempt to pull a single batch to verify shapes and labels
        images, labels = next(iter(train_loader))
        
        print("Batch successfully retrieved.")
        print(f"Batch Image Shape: {images.shape}") # Expected: [Batch, Channels, Height, Width]
        print(f"Batch Label Shape: {labels.shape}")
        print(f"Sample Labels: {labels[:5]}")
        print("\nData pipeline verification completed without errors. Task 1 is finished.")
        
    except FileNotFoundError:
        print(f"Error: 'data' directory not found at {base_data_path}. Ensure the dataset is organized.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")