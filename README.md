# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model
<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/3c66c126-b00f-4f46-a2a7-23ac704a7a2b" />
## DESIGN STEPS

### STEP 1:
Define the objective of classifying fashion items (T-shirts, trousers, dresses, shoes, etc.) using a Convolutional Neural Network (CNN).

### STEP 2:

Use the Fashion-MNIST dataset, which contains 60,000 training images and 10,000 test images of various clothing items.

### STEP 3:

Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

### STEP  4:
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers to extract features and classify clothing items.

### STEP 5:

Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs

### STEP 6:

Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

   
### STEP 7:

Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name: Niralya J 
### Register Number: 212224230188
```

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

```

```

# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```

```
# Train the Model
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

        print('Name: Niralya J')
        print('Register Number: 212224230188')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="387" height="283" alt="Screenshot 2026-03-09 114242" src="https://github.com/user-attachments/assets/bdba7fc5-5e60-401c-beb1-75c3e6d96195" />


### Confusion Matrix

<img width="1087" height="762" alt="image" src="https://github.com/user-attachments/assets/3659c976-cbfb-4db8-9a4c-d5ad7f3c0981" />


### Classification Report

<img width="580" height="432" alt="image" src="https://github.com/user-attachments/assets/aff70091-43f2-4c79-8f39-e445de2a94e1" />



### New Sample Data Prediction

<img width="598" height="651" alt="image" src="https://github.com/user-attachments/assets/4fd6e70c-0e03-4b1f-b417-1959b693022f" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.


