from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from encoding.parallel import DataParallelModel, DataParallelCriterion

import torchvision
from torchvision import transforms 

from sklearn.metrics.ranking import roc_auc_score
import re

import cv2
import math

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

        # pixData     = np.asarray(imageData)

        # Rchan = pixData[:,:,0]  # Red color channel
        # Gchan = pixData[:,:,1]  # Green color channel
        # Bchan = pixData[:,:,2]  # Blue color channel

        # Rchan_mean = Rchan.mean()
        # Gchan_mean = Gchan.mean()
        # Bchan_mean = Bchan.mean()

        # Rchan_sd = math.sqrt(Rchan.var())
        # Gchan_sd = math.sqrt(Gchan.var())
        # Bchan_sd = math.sqrt(Bchan.var())

        # normalize1 = transforms.Normalize([Rchan_mean,Gchan_mean,Bchan_mean], [Rchan_sd,Gchan_sd,Bchan_sd])

        normalize2 = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])

        # train
        transformList = []
        transformList.append(transforms.RandomResizedCrop(224))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        # transformList.append(normalize1)
        transformList.append(normalize2)
        transformSequence=transforms.Compose(transformList)


        # test
        # transformList = []
        # transformList.append(transforms.Resize(256))
        # transformList.append(transforms.TenCrop(224))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize1(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize2(crop) for crop in crops])))
        # transformSequence=transforms.Compose(transformList)



        imageLabel  = torch.FloatTensor(self.listImageLabels[index])
        # if self.transform != None : imageData = self.transform(imageData)
        if self.transform != None : imageData = transformSequence(imageData)
        return imageData, imageLabel
        
    def __len__(self):
        return len(self.listImagePaths)
    
# ======================================================= #

# print ('Testing the trained model')
def train ( pathDirData, pathFileTrain, pathFileVal, nnArchitecture, \
            nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, \
            transResize, transCrop, launchTimestamp, checkpoint ) :
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

    #-------------------- SETTINGS: NETWORK ARCHITECTURE
    if nnArchitecture   == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
    # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
    # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
    
    model = torch.nn.DataParallel(model).cuda()
    print("model init")
            
    #-------------------- SETTINGS: DATA TRANSFORMS
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
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
            
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=False)
    dataLoaderVal   = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=False)
    print("DATASET BUILDERS ... done")
    
    #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
    optimizer = optim.Adam (model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
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
        print("epoch :",epochID,"/",trMaxEpoch," | ","TimeStamp :",timestampSTART)
                        
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
        print("TRAINING ==== ")
        for batchID, (input_, target) in enumerate (dataLoader):
            print("\r","batchID", batchID,"/",len(dataLoader),end="")
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

def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        print("RUN TEST =====")
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda() 
        
        ########################################################
        modelCheckpoint = torch.load(pathModel)
        # model.load_state_dict(modelCheckpoint['state_dict'])
        

        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = modelCheckpoint['state_dict']
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)

        print("model loaded.")

        ########################################################

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)

        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=32, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()


        print(" Evaluate ...")
        for i, (input, target) in enumerate(dataLoaderTest):
            
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)
            
            bs, n_crops, c, h, w = input.size()
            
            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)
            
            out = model(varInput)
            outMean = out.view(bs, n_crops, -1).mean(1)
            
            outPRED = torch.cat((outPRED, outMean.data), 0)
            print("\r","===> ",i,"/",len(dataLoaderTest),end="")

        aurocIndividual = computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        return

def computeAUROC (dataGT, dataPRED, classCount):
    
    outAUROC = []
    
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        
    return outAUROC

# ==================================== #

if __name__ == "__main__" : 
    torch.cuda.empty_cache()
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    # Path to the directory with images
    pathDirData = '/home/memoming/study/CheXNet/database'
    # pathDirData = "/srv/repo/users/memoming/CheXNet/database"

    # images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain   = os.path.join('.','dataIndex','train_categorical.txt')
    pathFileVal     = os.path.join('.','dataIndex','val_categorical.txt')
    pathFileTest    = os.path.join('.','dataIndex','test_categorical.txt')

    # Neural network parameters: type of the network, is it pre-trained 
    # on imagenet, number of classes
    nnArchitecture  = DENSENET121
    nnIsTrained     = True
    nnClassCount    = 4

    # Training settings: batch size, maximum number of epochs
    trBatchSize     = 512 #origin : train&test : 16 / my : train : 256 -> 128 / test : 32
    trMaxEpoch      = 100 #100

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize  = 256
    imgtransCrop    = 224
        
    pathModel = 'model_categorical_-1_0_' + timestampLaunch + '.pth.tar'

    print ('Training NN architecture = ', nnArchitecture)
    train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)


    # pathModel = "m-10122019-182842.pth.tar"
    # pathModel = os.path.join("models","m-27112019-174526.pth.tar")
    # test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

# ========================================== #