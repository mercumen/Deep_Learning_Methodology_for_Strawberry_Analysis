## Project Structure

```text
strawberry_project/
├── data/               # Raw and processed images (Git ignored)
├── src/                # All Python source codes
│   ├── download_data.py
│   ├── organize_data.py
│   └── data_loader.py
├── experiments/        # Log files and model checkpoints
├── results/            # Performance charts and final tables
├── venv/               # Virtual environment (Git ignored)
├── requirements.txt    # Library dependencies
└── README.md           # Project instructions
```

## Project Overview

# Strawberry Ripeness Classification Project

This project focuses on classifying strawberries into three categories: Partially Ripe, Ripe, and Unripe. The goal is to demonstrate deep learning methodologies

## Prerequisites

Before starting, ensure you have Python installed on your system.

## Installation Steps

### 1. Virtual Environment Setup
It is mandatory to use a virtual environment to avoid library version conflicts.
- Open your terminal in the project folder.
- Create the environment:
  `python -m venv venv`
- Activate the environment:
  - Windows: `.\venv\Scripts\activate`
  - Mac/Linux: `source venv/bin/activate`

### 2. Install Required Libraries
Install all necessary dependencies using the requirements file:
`pip install -r requirements.txt`

### 3. Kaggle API Configuration
We use an automated script to fetch the dataset.
- Go to your Kaggle account settings and click "Create New API Token".
- Move the downloaded `kaggle.json` file to the `.kaggle` folder in your user directory:
  - Windows: `C:\Users\<YourUsername>\.kaggle\`
  - Mac/Linux: `~/.kaggle/`

## Data Pipeline Execution

Follow these steps in order to prepare the dataset for training:

### Step 1: Download the Dataset
Run the download script to fetch raw data:
`python src/download_data.py`

### Step 2: Organize and Split Data
Since the raw data is in YOLO format, run this script to convert it into a classification format and split it (80% Train, 20% Test):
`python src/organize_data.py`

### Step 3: Verify the Data Loader
Run the loader script to confirm that images are correctly resized to 128x128, normalized, and ready for training:
`python src/data_loader.py`

If you see the message "Test completed without errors," the data pipeline is ready for model training.

## Model Training (Task 2 — Baseline CNN)

After preparing the dataset, we implemented and trained a baseline Convolutional Neural Network (CNN) model for strawberry ripeness classification.

### Model Architecture

The model includes:
- Convolution layers
- ReLU activation functions
- MaxPooling layers
- Fully connected layers

The architecture is intentionally kept simple because this task is designed as a baseline experiment without regularization.

### Training Configuration

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 10

### Step 4: Train the Baseline CNN

Run the following command to train the baseline model:

`python src/train.py`

This script will:
- train the CNN for 10 epochs
- compute train and test loss per epoch
- compute train and test accuracy per epoch
- save the best model checkpoint
- save all experiment metrics in `experiments/baseline/`

### Results

- Final Test Accuracy: 48.25%
- Best Test Accuracy: 50.00%

### Observations

- Training accuracy increased from 42.48% to 60.62%, indicating that the model learned meaningful features from the data.
- Test accuracy fluctuated during training and reached a maximum of 50.00%.
- This behavior suggests limited generalization capability, which is expected for a baseline model without regularization.

### Output Files

All results are saved in:

`experiments/baseline/`

Files:
- best_model.pth → best performing baseline model
- metrics.txt → loss and accuracy values for each epoch
