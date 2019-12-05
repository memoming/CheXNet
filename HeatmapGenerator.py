import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import math
import re

from DenseNet import DenseNet121


from sklearn import preprocessing

#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator ():

 
    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):
       
        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True).cuda()
        # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True).cuda()
        # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True).cuda()
          
        model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(pathModel)

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
        # model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model.module.densenet121.features
        self.model.eval()
        
        #---- Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        #---- Load image, transform, convert 
        imageData   = Image.open(pathImageFile).convert('RGB')
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

        # normalize = transforms.Normalize([Rchan_mean,Gchan_mean,Bchan_mean], [Rchan_sd,Gchan_sd,Bchan_sd])
        # transformList = []
        # transformList.append(transforms.RandomResizedCrop(224))
        # transformList.append(transforms.RandomHorizontalFlip())
        # transformList.append(transforms.ToTensor())
        # transformList.append(normalize)      
        # self.transformSequence = transforms.Compose(transformList)


        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
        
        self.model.cuda()
        output = self.model(input.cuda())
        
        #---- Generate heatmap
        threshold = 0.9
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            eachWeight = self.weights[i]
            if i == 0: 
                heatmap = eachWeight * map
            else : 
                heatmap += eachWeight * map
        
        #---- Blend original and heatmap 
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = npHeatmap / np.max(npHeatmap)

        cam = cv2.resize(cam, (transCrop, transCrop))
        threshold = 0.8

        for i,eachList in enumerate(cam) :
            for j,each in enumerate(list(eachList)) :
                if each < threshold : cam[i][j] = None
        
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        img = heatmap * 0.5 + imgOriginal
        cv2.imwrite(pathOutputFile, img)
        
#-------------------------------------------------------------------------------- 

if __name__ == "__main__" :

    # pathDatasetFile    = os.path.join(".","dataIndex","test_1.txt")
    # pathImageDirectory = os.path.join(".","database")
    # listImagePaths     = []
    # listImageLabels    = []
    # fileDescriptor     = open(pathDatasetFile, "r")

    # line = True
    # while line:
    #     line = fileDescriptor.readline()
    #     if line:
    #         lineItems       = line.split()
    #         temp_path       = lineItems[0].split("/")
    #         lineItems[0]    = os.path.join(temp_path[0],temp_path[1])
    #         imagePath       = os.path.join(pathImageDirectory, lineItems[0])
    #         imageLabel      = lineItems[1:]
    #         imageLabel      = [int(i) for i in imageLabel]
    #         listImagePaths.append(imagePath)
    #         listImageLabels.append(imageLabel)   
    # fileDescriptor.close()

    nnArchitecture  = 'DENSE-NET-121'
    nnClassCount    = 14
    transCrop       = 224
    pathModel       = os.path.join(".","models","m-04122019-124803.pth.tar")
    # pathModel       = os.path.join(".","models","m-25012018-123527.pth.tar")
    heatmapGen      = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
    print("Generator Loaded.")

    pathInputImage = os.path.join("test","00009285_000.png")
    pathOutputImage = os.path.join("test","heatmap_threshold_0.8.png")
    heatmapGen.generate(pathInputImage, pathOutputImage, transCrop)
    print("Done !")


    # for i,eachImagePath in enumerate(listImagePaths) :
    #     pathSaveDir     = os.path.join("/mnt","Hdd1","memoming","cheXNet","analyze","test_1_to_heatmap")
    #     pathInputImage  = eachImagePath
    #     pathOutputImage = os.path.join(pathSaveDir,eachImagePath.split(os.sep)[-1])
    #     heatmapGen.generate(pathInputImage, pathOutputImage, transCrop)
    #     print("\r","Generate ==> ",eachImagePath.split(os.sep)[-1],"(",i,"/",len(listImagePaths),")",end="")

