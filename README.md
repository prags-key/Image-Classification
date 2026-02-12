# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The objective of this project is to create a CNN that can categorize images of fashion items from the Fashion MNIST dataset. This dataset includes grayscale images of clothing and accessories such as T-shirts, trousers, dresses, and footwear. The task is to accurately predict the correct category for each image while ensuring the model is efficient and robust.

1.Training data: 60,000 images

2.Test data: 10,000 images

3.Classes: 10 fashion categories

The CNN consists of multiple convolutional layers with activation functions, followed by pooling layers, and ends with fully connected layers to output predictions for all 10 categories.

## Neural Network Model

<img width="1037" height="406" alt="image" src="https://github.com/user-attachments/assets/859a0f24-16ad-43d6-ad53-b6ed819ceb81" />


## DESIGN STEPS

### STEP 1:

Import the necessary libraries such as NumPy, Matplotlib, and PyTorch.

### STEP 2:

Load and preprocess the dataset:

Resize images to a fixed size (128×128). Normalize pixel values to a range between 0 and 1. Convert labels into numerical format if necessary.

### STEP 3:

Define the CNN Architecture, which includes:

Input Layer: Shape (8,128,128) Convolutional Layer 1: 8 filters, kernel size (16×16), ReLU activation Max-Pooling Layer 1: Pool size (2×2) Convolutional Layer 2: 24 filters, kernel size (8×8), ReLU activation Max-Pooling Layer 2: Pool size (2×2) Fully Connected (Dense) Layer: First Dense Layer with 256 neurons Second Dense Layer with 128 neurons Output Layer for classification

### STEP 4:

Define the loss function (e.g., Cross-Entropy Loss for classification) and optimizer (e.g., Adam or SGD).

### STEP 5:
Train the model by passing training data through the network, calculating the loss, and updating the weights using backpropagation.


### STEP 6:
Evaluate the trained model on the test dataset using accuracy, confusion matrix, and other performance metrics.


### STEP 7:

Make predictions on new images and analyze the results.

## PROGRAM

### NAME: PRAGATHI KUMAR

### REGISTER NUMBER: 212224230200

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 → 14x14
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 → 7x7

        x = x.view(x.size(0), -1)                  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x





```

```
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```
from torch.utils.data import DataLoader, TensorDataset

# Dummy data (just to test output)
images = torch.randn(64, 1, 28, 28)   # 64 fake images
labels = torch.randint(0, 10, (64,)) # 64 fake labels

train_dataset = TensorDataset(images, labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


```

```
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: PRAGATHI KUMAR')
        print('Register Number: 212224230200')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

## OUTPUT
### Training Loss per Epoch

<img width="562" height="263" alt="image" src="https://github.com/user-attachments/assets/7aa74ec3-91e1-498d-a522-5b260fe6e905" />


### Confusion Matrix

<img width="906" height="760" alt="image" src="https://github.com/user-attachments/assets/f60ea125-816a-4441-ae63-c48cca7e6838" />


### Classification Report

<img width="650" height="414" alt="image" src="https://github.com/user-attachments/assets/779a1b5d-c52c-4268-8c58-4bc74e401ea7" />



### New Sample Data Prediction
<img width="703" height="603" alt="image" src="https://github.com/user-attachments/assets/134d9645-3f47-4dc2-9a35-bbca5d762b93" />



## RESULT
Thus, a convolutional neural network for image classification was successfully implemented and verified using an Excel-based dataset
