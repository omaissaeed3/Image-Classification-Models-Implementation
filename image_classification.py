import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define dataset transformation (normalize and convert to tensors)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define labels subset (3 classes)
selected_classes = [0, 1, 2]  # Example: Airplane, Automobile, Bird

# Filter dataset
trainset.data = trainset.data[[i for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]]
trainset.targets = [trainset.targets[i] for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]

testset.data = testset.data[[i for i in range(len(testset.targets)) if testset.targets[i] in selected_classes]]
testset.targets = [testset.targets[i] for i in range(len(testset.targets)) if testset.targets[i] in selected_classes]

# Display a sample image
plt.imshow(trainset.data[0])
plt.title(f"Sample Image - Class {trainset.targets[0]}")
plt.axis("off")
plt.show()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Flatten images for SVM (from 32x32x3 to 1D array)
X_train = trainset.data.reshape(len(trainset.data), -1)[:2000]
y_train = trainset.targets[:2000]

X_test = testset.data.reshape(len(testset.data), -1)[:2000]
y_test = testset.targets[:2000]

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print(f"SVM Accuracy: {svm_accuracy:.4f}")