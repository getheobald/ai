import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json

'''
In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.
'''


'''
PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.
'''

transform = transforms.Compose([                            # Use transforms to convert images to tensors and normalize them
    transforms.ToTensor(),                                  # convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])             # Common method for grayscale images
])

# using same as lab
batch_size = 64


'''
PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.
'''

trainset = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False) # shuffle false to make plotting easier


'''
PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.
'''

feedforward_net = FF_Net()
conv_net = Conv_Net()



'''
PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.
'''

criterion = nn.CrossEntropyLoss()

optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001)

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Adam is a bit more sophisticated than SGD because it has an adaptive learning rate so converges faster
# need to pass in parameters because that's what we're optimizing
# 0.001 is the standard/default learning rate for Adam



'''
PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)
'''

num_epochs_ffn = 10  # can adjust to improve accuracy
loss_accumulation_ffn = []

for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
    running_loss_ffn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Flatten inputs for ffn - handled inside model

        # zero the parameter gradients
        optimizer_ffn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ffn.step()
        running_loss_ffn += loss.item()

    loss_accumulation_ffn.append(running_loss_ffn)
    print(f"Training loss: {running_loss_ffn}")

print('Finished Training')

torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)


num_epochs_cnn = 10  # can adjust to improve accuracy
loss_accumulation_cnn = []

for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
    running_loss_cnn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer_cnn.zero_grad()

        # forward + backward + optimize
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()

    loss_accumulation_cnn.append(running_loss_cnn)
    print(f"Training loss: {running_loss_cnn}")

print('Finished Training')

torch.save(conv_net.state_dict(), 'cnn.pth')  # Saves model file (upload with submission)


# save json outputs
with open('loss_history.json', 'w') as f:
    json.dump({'ffn': loss_accumulation_ffn, 'cnn': loss_accumulation_cnn}, f)

'''
PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.


For each batch of test data, eval loop needs to:
- Run test data through both models to get predictions
- Figure out which class each model predicted (highest value in the tensor)
- Compare predictions to true lables
- Keep a running count of correct predictions and total images
'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

correct_ffn = 0
total_ffn = 0

correct_cnn = 0
total_cnn = 0

with torch.no_grad():           # since we're not training, we don't need to calculate the gradients for our outputs
    for data in testloader:
        inputs, labels = data
        
        pred_ffn = feedforward_net(inputs)
        pred_cnn = conv_net(inputs)

        total_ffn += labels.size(0)
        correct_ffn += (pred_ffn.argmax(1) == labels).type(torch.float).sum().item()

        total_cnn += labels.size(0)
        correct_cnn += (pred_cnn.argmax(1) == labels).type(torch.float).sum().item()


print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
print('Accuracy for convolutional network: ', correct_cnn/total_cnn)


'''
PART 7:
Generate plots
'''

# fashion mnist data set class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# output one image classified correctly and one classified incorrectly by each model, with labels
def plot_predictions(model, model_name, testloader, class_names):
    found_correct = False
    found_incorrect = False
    correct_img = correct_pred = correct_true = None
    incorrect_img = incorrect_pred = incorrect_true = None

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            preds = outputs.argmax(1)

            for j in range(len(labels)):
                if not found_correct and preds[j] == labels[j]:
                    correct_img = inputs[j]
                    correct_pred = preds[j].item()
                    correct_true = labels[j].item()
                    found_correct = True
                if not found_incorrect and preds[j] != labels[j]:
                    incorrect_img = inputs[j]
                    incorrect_pred = preds[j].item()
                    incorrect_true = labels[j].item()
                    found_incorrect = True
            if found_correct and found_incorrect:
                break

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(f'{model_name} Predictions')

    axes[0].imshow(correct_img.squeeze(), cmap='gray')
    axes[0].set_title(f'Correct\nPred: {class_names[correct_pred]}\nTrue: {class_names[correct_true]}')
    axes[0].axis('off')

    axes[1].imshow(incorrect_img.squeeze(), cmap='gray')
    axes[1].set_title(f'Incorrect\nPred: {class_names[incorrect_pred]}\nTrue: {class_names[incorrect_true]}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_predictions.png')
    plt.show()

plot_predictions(feedforward_net, 'FFN', testloader, class_names)
plot_predictions(conv_net, 'CNN', testloader, class_names)


# plot training loss over time for ffn
plt.figure()
plt.plot(range(1, num_epochs_ffn + 1), loss_accumulation_ffn)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('FFN Training Loss')
plt.savefig('ffn_loss.png')
plt.show()


# plot training loss over time for cnn
plt.figure()
plt.plot(range(1, num_epochs_cnn + 1), loss_accumulation_cnn)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training Loss')
plt.savefig('cnn_loss.png')
plt.show()


# count and display parameters for each nn
ffn_params = sum(p.numel() for p in feedforward_net.parameters())
cnn_params = sum(p.numel() for p in conv_net.parameters())
print(f'FFN total parameters: {ffn_params}')
print(f'CNN total parameters: {cnn_params}')


# confusion matrices
def plot_confusion_matrix(model, model_name, testloader, class_names):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    plt.show()

plot_confusion_matrix(feedforward_net, 'FFN', testloader, class_names)
plot_confusion_matrix(conv_net, 'CNN', testloader, class_names)


'''
PART 8:
Compare the performance and characteristics of FFN and CNN models.
'''
# see report