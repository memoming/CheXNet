from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import time
import sys

import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms 

# ======================================================= #

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

# ======================================================= #

class DatasetGenerator (Dataset):
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths     = []
        self.listImageLabels    = []
        self.transform          = transform
        fileDescriptor          = open(pathDatasetFile, "r")

        line = True
        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems       = line.split()
                temp_path       = lineItems[0].split("/")
                lineItems[0]    = os.path.join(temp_path[0],temp_path[1])
                imagePath       = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel      = lineItems[1:]
                imageLabel      = [int(i) for i in imageLabel]
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
        fileDescriptor.close()
    
    
    def __getitem__(self, index):
        imagePath   = self.listImagePaths[index]
        imageData   = Image.open(imagePath).convert('RGB')
        imageLabel  = torch.FloatTensor(self.listImageLabels[index])
        if self.transform != None : imageData = self.transform(imageData)
        return imageData, imageLabel
        
    def __len__(self):
        return len(self.listImagePaths)
    
# ======================================================= #

# print ('Testing the trained model')
def train ( pathDirData, pathFileTrain, pathFileVal, nnArchitecture, \
            nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, \
            transResize, transCrop, launchTimestamp, checkpoint ) :

    #-------------------- SETTINGS: NETWORK ARCHITECTURE
    if nnArchitecture   == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
    # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
    # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
    
    model = torch.nn.DataParallel(model).cuda()
    print("model init")
            
    #-------------------- SETTINGS: DATA TRANSFORMS
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformList = []
    transformList.append(transforms.RandomResizedCrop(transCrop))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)      
    transformSequence=transforms.Compose(transformList)
    print("DATA TRANSFORMS ... done")

    #-------------------- SETTINGS: DATASET BUILDERS
    datasetTrain    = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
    datasetVal      = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
            
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=8, pin_memory=True)
    dataLoaderVal   = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=True)
    print("DATASET BUILDERS ... done")
    
    #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
    optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
    print("OPTIMIZER & SCHEDULER ... set")
            
    #-------------------- SETTINGS: LOSS
    loss = torch.nn.BCELoss(size_average = True)
    print("LOSS FUNC ... set")
    
    #---- Load checkpoint 
    if checkpoint != None:
        modelCheckpoint = torch.load(checkpoint)
        model.load_state_dict(modelCheckpoint['state_dict'])
        optimizer.load_state_dict(modelCheckpoint['optimizer'])

    
    #---- TRAIN THE NETWORK
    print("START TRAIN")
    
    lossMIN = 100000
    
    for epochID in range (0, trMaxEpoch):
        
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampSTART = timestampDate + '-' + timestampTime
        print("epoch_",epochID,"TimeStamp :",timestampSTART)
                        
        epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
        lossVal, losstensor = epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
        
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampEND = timestampDate + '-' + timestampTime
        
        scheduler.step(losstensor.item())
        
        if lossVal < lossMIN:
            lossMIN = lossVal    
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
            print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
        else:
            print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                     
# ======================================================= #

def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        print("Enter Epoch Train")
        for batchID, (input_, target) in enumerate (dataLoader):
            print("barchID", batchID,"/",len(dataLoader))
            target = target.cuda()
                 
            varInput = torch.autograd.Variable(input_)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)
            
            lossvalue = loss(varOutput, varTarget)
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

# ======================================================= #

def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        lossVal         = 0
        lossValNorm     = 0
        losstensorMean  = 0
        
        for i, (input_, target) in enumerate (dataLoader):
            target      = target.cuda()
            with torch.no_grad() :
                varInput    = torch.autograd.Variable(input_)
                varTarget   = torch.autograd.Variable(target)
                    
                varOutput   = model(varInput)
                losstensor  = loss(varOutput, varTarget)
                losstensorMean  += losstensor
                lossVal         += losstensor.item()
                lossValNorm     += 1

        outLoss         = lossVal / lossValNorm
        losstensorMean  = losstensorMean / lossValNorm
        return outLoss, losstensorMean

# ======================================================= #

if __name__ == "__main__" : 
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    # Path to the directory with images
    pathDirData = '/srv/repo/users/memoming/CheXNet/database'

    # images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain   = os.path.join('.','dataIndex','train_1.txt')
    pathFileVal     = os.path.join('.','dataIndex','val_1.txt')
    pathFileTest    = os.path.join('.','dataIndex','test_1.txt')

    # Neural network parameters: type of the network, is it pre-trained 
    # on imagenet, number of classes
    nnArchitecture  = DENSENET121
    nnIsTrained     = True
    nnClassCount    = 14

    # Training settings: batch size, maximum number of epochs
    trBatchSize     = 8 #16
    trMaxEpoch      = 10 #100

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize  = 256
    imgtransCrop    = 224
        
    pathModel = 'model_' + timestampLaunch + '.pth.tar'

    print ('Training NN architecture = ', nnArchitecture)
    train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)

