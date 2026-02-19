from unicodedata import name
from xml.parsers.expat import model
import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import math

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
# Have to define the dict out of a function so both main and the hook function can access it
# feature_maps = {}

def main ():
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    train_loader, valid_loader, test_loader, classes, batch_size = data_loader()
    visualize_batch(train_loader, classes)

    # create a complete CNN
    model = Net()
    print(model)
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    #########################################
    # Decide on Loss Function and Optimizer #
    #########################################
    '''
    Decide on a loss and optimization function that is best suited for this classification task. The linked code examples from above, 
    may be a good starting point; this PyTorch classification example Pay close attention to the value for learning rate as this value 
    determines how your model converges to a small error.
    The following is working code, but you can make your own adjustments.
    TODO: try to compare with ADAM optimizer
    '''

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # TODO, compare with optimizer ADAM 
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare hooks to capture feature maps for second part of assignment:
#     hook_handle_1 = model.conv1.register_forward_hook(feature_hook('conv1'))
#     hook_handle_2 = model.conv2.register_forward_hook(feature_hook('conv2'))    

    # Since we are analyzing RELU's effect on the middle feature maps, have to apply it here
#     feature_maps['conv2'] = F.relu(feature_maps['conv2'])

    # Remove hooks before training to not analyze every batch
#     hook_handle_1.remove()
#     hook_handle_2.remove()

    train(model, train_loader, valid_loader, criterion, optimizer, train_on_gpu)

    ###########################################################
    # Load the Model with the Lowest Validation Loss and Test #
    ###########################################################

    model.load_state_dict(torch.load('model_trained.pt'))

    test(model, test_loader, classes, batch_size, criterion, train_on_gpu)

    # Extract feauture maps:
    # conv1_feature_maps = self.conv1(x)

    # for i in range(8):
    #     plt.imshow(conv1_feature_maps[0, i].cpu(), cmap='gray')

    # Extract feauture maps:
    # conv2_feature_maps = self.conv2(x)

    # for i in range(8):
    #     plt.imshow(conv2_feature_maps[0, i].cpu(), cmap='gray')




##################################
# Data Loading and Preprocessing #
##################################
def data_loader():
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 5
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                                download=True, transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, valid_loader, test_loader, classes, batch_size

######################################
# Visualize a Batch of Training Data #
######################################
# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# def feature_hook(name):
#    def hook(module, input, output):
#       feature_maps[name] = output.detach()
#    return hook

def visualize_batch(train_loader, classes):
    # matplotlib inline

    # obtain one batch of training images
    dataiter = iter(train_loader)
    #images, labels = dataiter.next() #python, torchvision version match issue
    images, labels = next(dataiter)

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy() # convert images to numpy for display
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy() # convert labels to numpy for display

    num_images = images.shape[0]
    num_show = min(num_images, 20)
    num_columns = math.ceil(num_show / 2)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in range(num_show):
        ax = fig.add_subplot(2, num_columns, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])

    #############################
    # View image in more detail #
    #############################

    rgb_img = np.squeeze(images[3])
    channels = ['red channel', 'green channel', 'blue channel']

    fig = plt.figure(figsize = (36, 36)) 
    for idx in np.arange(rgb_img.shape[0]):
        ax = fig.add_subplot(1, 3, idx + 1)
        img = rgb_img[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(channels[idx])
        width, height = img.shape
        thresh = img.max()/2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y],2) if img[x][y] !=0 else 0
                ax.annotate(str(val), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center', size=8,
                        color='white' if img[x][y]<thresh else 'black')

########################
# My code starts here: #
########################
'''
Build up your own Convolutional Neural Network using Pytorch API:

nn.Conv2d(): for convolution
nn.MaxPool2d(): for maxpooling (spatial resolution reduction)
nn.Linear(): for last 1 or 2 layers of fully connected layer before the output layer.
nn.Dropout(): optional, dropout can be used to avoid overfitting.
F.relu(): Use ReLU as the activation function for all the hidden layers
The following is a skeleton example that's not completely working.
'''

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # TODO: Build multiple convolutional layers (sees 32x32x3 image tensor in the first hidden layer)
        # for example, conv1, conv2 and conv3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # TODO: Build some linear layers (fully connected)
        # for example, fc1 and fc2  
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=10)

        # TODO: dropout layer (p=0.25, you can adjust)
        # example self.dropout = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        # assume we have 3 convolutional layers defined above
        # and we do a maxpooling after each conv layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # TODO: flatten x at this point to get it ready to feed into the fully connected layer(s)
        # Can use this but need to figure out the actual value for a, b and c
        x = x.view(-1, 64 * 4 * 4)

        # optional add dropout layer
        #x = self.dropout(x)
        x = self.dropout(x)
        
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # optional add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x





###################################
# Network Training and Validation #
###################################
def train(model, train_loader, valid_loader, criterion, optimizer, train_on_gpu):
    # number of epochs to train the model, you decide the number
    n_epochs = 15

    valid_loss_min = np.inf # track change in validation loss

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model_trained.pt')
            valid_loss_min = valid_loss


##############################################
# Test the Trained Model on the Test Dataset #
##############################################
def test(model, test_loader, classes, batch_size, criterion, train_on_gpu):
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    # iterate over test data
    for batch_idx, (data, target) in enumerate(test_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    #################################
    # Visualize Sample Test Results #
    #################################

    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images.numpy()

    # move model inputs to cuda, if GPU available
    if train_on_gpu:
        images = images.cuda()

    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

    num_images = images.shape[0]
    num_show = min(num_images, 20)
    num_columns = math.ceil(num_show / 2)

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(num_show):
        ax = fig.add_subplot(2, num_columns, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))


if __name__=='__main__':
  main()