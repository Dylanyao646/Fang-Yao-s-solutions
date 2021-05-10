#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import glob
import cv2
import random
import torch.nn as nn
import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

 
training_path = 'C:/Users/Dylan/Downloads/digits/data/mnist/training'
test_path = 'C:/Users/Dylan/Downloads/digits/data/mnist/testing'
custom_path = 'C:/Users/Dylan/Downloads/digits/data/custom_set'

x_train=np.empty((0,1,28,28), dtype=int)
y_train=np.empty(0, dtype=int)
count=0
for i in range(0,10):
    folderpath=training_path + '/' + str(i)+'/*.png'
    print(folderpath)
    for filename in glob.iglob(folderpath, recursive=False):
        # Do something
        print('Loading ' + filename)
        im = cv2.imread(filename,0)
        x_train=np.append(x_train, im.reshape(1,1,28,28), axis=0)
        y_train=np.append(y_train, [i], axis=0)
        print('Image loaded:' + str(count))
        count=count+1
print(x_train)
print(x_train.shape)
print(y_train)
print(y_train.shape)
    

x_test=np.empty((0,1,28,28), dtype=int)
y_test=np.empty(0, dtype=int)
count=0
for i in range(0,10):
    folderpath=test_path + '/' + str(i)+'/*.png'
    print(folderpath)
    for filename in glob.iglob(folderpath, recursive=False):
        # Do something
        print('Loading ' + filename)
        im = cv2.imread(filename,0)
        x_test=np.append(x_test, im.reshape(1,1,28,28), axis=0)
        y_test=np.append(y_test, [i], axis=0)
        print('Image loaded:' + str(count))
        count=count+1
print(x_test)
print(x_test.shape)
print(y_test)
print(y_test.shape)
    

x_custom=np.empty((0,1,50,50), dtype=int)
y_custom=np.empty(0, dtype=int)
count=0
for i in range(0,10):
    folderpath=imgpath + '/' + str(i)+'/*.jpg'
    print(folderpath)
    for filename in glob.iglob(folderpath, recursive=False):
        # Do something
        print('Loading ' + filename)
        im = cv2.imread(filename,0)
        x_custom=np.append(x_custom, im.reshape(1,1,50,50), axis=0)
        y_custom=np.append(y_custom, [i], axis=0)
        print('Image loaded:' + str(count))
        count=count+1
print(x_custom)
print(x_custom.shape)
print(y_custom)
print(y_custom.shape)

batch_size = 64
validation_split = .2
shuffle_dataset = True
random_seed= 42

datasets_x_train = torch.from_numpy(x_train).float()
datasets_y_train = torch.from_numpy(y_train).long()

dataset_train = dataf.TensorDataset(datasets_x_train,datasets_y_train)
#train_loader = dataf.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

datasets_x_test = torch.from_numpy(x_test).float()
datasets_y_test = torch.from_numpy(y_test).long()

dataset_test = dataf.TensorDataset(datasets_x_test,datasets_y_test)
test_loader = dataf.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)


# Creating data indices for training and validation splits:
dataset_size = len(dataset_train)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                sampler=valid_sampler)


# In[41]:


pip install tensorboard


# In[42]:



## define th model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 5 * 5, 10) 

    def forward(self, x):
        # Set 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        # Set 2
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        #Flatten
        out = out.view(out.size(0), -1)

        #Dense
        out = self.fc1(out)
        return out
    
# instantiate the CNN
model_scratch = CNN()

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
    
    


# In[43]:


#####################################################
## Create a CNN to Classify MNIST (from Scratch) ###
#####################################################

### select loss function
criterion_scratch = nn.CrossEntropyLoss()

### select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.01) # use SGD to get more accuracy
#optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.01) # use Adam to improve the training speed

# create dictionary for all loaders in one
loaders_scratch = {}
loaders_scratch['train'] = train_loader
loaders_scratch['valid'] = validation_loader
loaders_scratch['test'] = test_loader


# In[44]:


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                data = data.view(64,1,28,28)
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model_scratch(data)
            # calculate the batch loss
            loss = criterion_scratch(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update average training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            #train_loss += loss.item()*data.size(0)
        
        ######################    
        # validate the model #
        ######################
            
            
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model_scratch(data)
            # calculate the batch loss
            loss = criterion_scratch(output, target)
            # update average validation loss 
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            #valid_loss += loss.item()*data.size(0)
    

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                  valid_loss_min,
                  valid_loss))
            torch.save(model.state_dict(), 'model_scratch.pt')
            valid_loss_min = valid_loss    
    # return trained model
    return model


# train the model
model_scratch = train(30, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))


# In[45]:


###### Test the Model ##########

def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)


# In[ ]:




