import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
In this file you will write the model definition for a convolutional neural network. 

Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, and convolutional layers in PyTorch:

    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


Whether you need to normalize outputs using softmax depends on your choice of loss function. PyTorch documentation is available at
https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton requires normalized outputs or not.
'''

class Conv_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(64*11*11, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        post_conv = self.conv_stack(x)
        post_flatten = self.flatten(post_conv)
        post_linear = self.linear_stack(post_flatten)
        return post_linear
        

'''
Convolutional NNs compared to feed-forward NNs
	• FFN flattens immediately
	• CNN slides small filters (kernels) over the image to detect local patterns like edges and textures before eventually flattening and classifying
	• CNNs tend to do better on images

CNNs need a new convolutional layer:
Nn.Conv2d(in_channels, out_channels, kernel_size)
	• For grayscale image, in_channels=1
	• For color (RGB), in_channels=3
	• Out_channels is how many kernels you want
		○ Common to double kernels in second layer because A dimensions shrink through pooling and B early layers detect simple features but later layers detect more complex combinations and benefit from more kernels
		○ 32 and 64 are common starting choices. Could also use 16 and 32, 64 and 128.
	• Kernel_size is kow big each filter is
        Kernel_size=3 means a 3x3 filter

Nn.MaxPool2d(kernel_size) downsamples the image by taking the max value in each window
	• Kernel_size = 2 halves spatial dimentions
	• Reduces computation and makes network more robust to small shifts in the image
	• Doesn't do any learning, it's just a tool to reduce spatial size
	• More pools = faster training, but can lose information

Typical CNN struture:
Conv --> ReLU --> MaxPool --> Conv --> ReLU --> MaxPool --> Flatten --> Linear --> ReLU --> Linear

Note: you can also do conv pool relu instead of conv relu pool for the conv layers, but latter is preferred because it's better data utilization and just the standard architecture

Challenge is knowing what size to pass to first linear layer after flattening because conv and pool layers change spatial dimensions

Conv2d with kernel_size=3 shrinks the image by 2 in each dimension (i.e. 28 to 26)

MaxPool2d(2) halves it (26 to 13)

How to pick num layers
	• Informed trial and error
	• For this, more than 2 conv blocks doesn't make sense because that would bring us to 4x4 which is very little to extract features from
	• One hidden layer is usually enough for simple classification once the conv layers have already done the feature extraction work, and adding more linear layers after a CNN gives diminishing returns
	• Start simple and add complexity only if accuracy is insufficient
	• More layers = more params = longer training & risk of overfitting on small datasets

First resource notes
	• Specialized for being able to detect patterns
	• Convolutional layers are just like other layers in that they receive input, transform it, and produce outputs
	• But the transformation is a convolution operation
	• Patterns = edges, shapes, textures, curves, objects, colors, etc.
	• When adding a conv layer, you have to specify how many filters the layer has. Number of filters determines number of output channels.
	• Filter is a small matrix (tensor) and you decide num rows/cols, and values within it are init random
	• Pattern detectors are derived automatically by the network
	• Filter values start out random, and values change as network learns during training

Sizes for this CNN
	• Take in 28*28
	• After first conv we have 26*26
	• Relu
	• After first maxPool we have 13*13
	• Conv, 11*11
	• Relu
	• maxPool 5*5 because it rounds down
		○ Nvm, should probably skip this because that's quite small
	• Hidden linear layer size should be somewhere between input size and output size, but again it's mostly convention and trial & error
'''