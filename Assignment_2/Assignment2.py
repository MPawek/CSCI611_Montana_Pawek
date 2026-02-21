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
# Originally made a dict as I was thinking of using different convolutional layers, but decided to only use the first. Kept the dict for simplicity
feature_maps = {}

def main ():
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    # Use data loader to load the CIFAR10 dataset, and visualize a batch of training data
    train_loader, valid_loader, test_loader, classes, batch_size = data_loader()
    
    # Visualize_batch used as debugging tool, but is not necessary for the model to run
    # visualize_batch(train_loader, classes)

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
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    # TODO, compare with optimizer ADAM 
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model, returning training and validation losses for each epoch to plot later, as well as the feature maps for the last batch of training and validation data to analyze later
    train_losses, valid_losses = train(model, train_loader, valid_loader, criterion, optimizer, train_on_gpu)

    ###########################################################
    # Load the Model with the Lowest Validation Loss and Test #
    ###########################################################

    model.load_state_dict(torch.load('model_trained.pt'))

    test(model, test_loader, classes, batch_size, criterion, train_on_gpu)

    # Part 2 of assignment: Visualize and Analyze Feature Maps
    get_feature_maps(model, test_loader, classes, train_on_gpu)

    # Plot training and validation loss over all epochs
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.grid(True)
    plt.legend()

    plt.show()

def get_feature_maps(model, test_loader, classes, train_on_gpu):
    device = torch.device("cuda" if train_on_gpu else "cpu")
    # Set to eval so we don't change parameters/use dropout
    model.eval()

    # Need to get 3 different classes for images
    images_to_analyze = {}
    for images, labels in test_loader:
        for i in range(images.size(0)):
            y = int(labels[i].item())
            if y not in images_to_analyze:
                images_to_analyze[y] = (images[i], y)
                if len(images_to_analyze) == 3:
                    break
        if len(images_to_analyze) == 3:
            break

    # We only need conv1 maps for each image, so that's where we'll put the hook
    handle = model.conv1.register_forward_hook(feature_hook('conv1'))

    # We don't need gradients for visualization, so turn off for efficiency
    torch.set_grad_enabled(False)


    for (image_tensor, class_index) in images_to_analyze.values():
        # Prepare single-image batch
        x = image_tensor.unsqueeze(0).to(device)

        # Forward pass triggers hook and fills feature_maps['conv1'], we don't need the output for this
        _ = model(x)

        # Apply ReLU here
        conv1_maps = F.relu(feature_maps['conv1'])

        # Got an error when I set range(8) for the loop below, have to make sure we do the minimum of total feature maps or 8
        features = conv1_maps.shape[1]
        n_show = min(8, features)

        # Formatted plots below with ChatGPT
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(f"Class: {classes[y]} | Layer: conv1 | Showing {n_show} feature maps", fontsize=14)

        # Original image (left)
        ax0 = plt.subplot2grid((2, 5), (0, 0), rowspan=2)
        imshow(image_tensor)
        ax0.set_title("Input image")
        ax0.axis("off")

        # Feature maps (right): 2x4 grid
        fmap_chw = conv1_maps[0].detach().cpu()
        for ch in range(n_show):
            r = ch // 4
            c = (ch % 4) + 1
            ax = plt.subplot2grid((2, 5), (r, c))

            m = fmap_chw[ch].numpy()
            # Normalize each channel to [0,1] for display
            m = m - m.min()
            if m.max() > 0:
                m = m / m.max()

            ax.imshow(m, cmap="gray")
            ax.set_title(f"conv1 ch {ch}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    handle.remove()

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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, valid_loader, test_loader, classes, batch_size

######################################
# Visualize a Batch of Training Data #
######################################
# helper function to un-normalize and display an image
# When using CUDA, the images are on the GPU, so we need to move them back to the CPU before converting to numpy for display
def imshow(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def feature_hook(name):
    def hook(module, input, output):
       feature_maps[name] = output.detach()
    return hook

def visualize_batch(train_loader, classes):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    #images, labels = dataiter.next() #python, torchvision version match issue
    images, labels = next(dataiter)

    # move model inputs to CUDA, if GPU available
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy() # convert images to numpy for display
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy() # convert labels to numpy for display

    # Calculations to ensure we don't go out of bounds when using the images object
    num_images = images.shape[0]
    num_show = min(num_images, 20)
    num_columns = math.ceil(num_show / 2)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    # Ran into an out of bounds error when trying to always display all 5 images in the batch, so we have to do the math to make sure the range value is not too large
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
        # 3 Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # TODO: Build some linear layers (fully connected)
        # for example, fc1 and fc2  
        # 2 Linear layers
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

        # TODO: dropout layer (p=0.25, you can adjust)
        # example self.dropout = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        # assume we have 3 convolutional layers defined above
        # and we do a maxpooling after each conv layer
        # Separate the first convolutional layer so we can return the output for feauture map visualization and analysis
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # TODO: flatten x at this point to get it ready to feed into the fully connected layer(s)
        x = x.view(-1, 64 * 4 * 4)
        
        # dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)

        # Return x1 for visualization and analysis of the feature maps after the first convolutional layer, and x for the final output
        return x





###################################
# Network Training and Validation #
###################################
def train(model, train_loader, valid_loader, criterion, optimizer, train_on_gpu):
    # number of epochs to train the model, you decide the number
    n_epochs = 20

    valid_loss_min = np.inf # track change in validation loss

    # Arrays to hold training and validation losses for each epoch
    train_losses = []
    valid_losses = []

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
            
        # Append calculated average losses for this epoch to the arrays
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

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

    # Return the values we tracked for data analysis and visualization later
    return train_losses, valid_losses

##############################################
# Test the Trained Model on the Test Dataset #
##############################################
def test(model, test_loader, classes, batch_size, criterion, train_on_gpu):
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # Code to find top images for 3 filters
    layer_name = 'conv1'
    selected_filters = [0, 4, 5]
    top_k = 5
    top_images = {f: [] for f in selected_filters}

    model.eval()

    # Add hook for analysis
    layer_modue = getattr(model, layer_name)
    hook_handle = layer_modue.register_forward_hook(feature_hook(layer_name))
    cur_index = 0

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

        current_batch_size = data.size(0)

        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

        # Remember to apply ReLU for analysis
        fm = feature_maps[layer_name]
        fm = F.relu(fm)

        for f in selected_filters:
            scores = fm[:, f, :, :].amax(dim=(1, 2))

            for i in range(current_batch_size):
                act = float(scores[i].item())
                image_tensor = data[i].detach().cpu()
                label = int(target[i].detach().cpu().item())
                index = cur_index + i

                top_images[f].append((act, image_tensor, label, index))
                top_images[f].sort(key=lambda x: x[0], reverse=True)
                if len(top_images[f]) > top_k:
                    top_images[f] = top_images[f][:top_k]

        cur_index += current_batch_size
    
    hook_handle.remove()

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

    # Show the data we got for the top images for each filter
    # ChatGPT used to help with formatting
    fig = plt.figure(figsize=(15, 8))

    for row, f in enumerate(selected_filters):
        for col, (act, img_tensor, label, index) in enumerate(top_images[f]):
            ax = fig.add_subplot(len(selected_filters), top_k, row * top_k + col + 1)
            img = img_tensor.numpy()
            img = np.transpose(img, (1, 2, 0)) 
            img = img / 2 + 0.5
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Act={act:.2f}\n{classes[label]} ch={f}', fontsize=8)

    plt.suptitle(f"Top {top_k} Maximally Activating Images ({layer_name} - Max activation, after ReLU)")
    plt.tight_layout()
    plt.show()


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