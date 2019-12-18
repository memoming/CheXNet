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
import random

from DenseNet import DenseNet121


from sklearn import preprocessing
from torchsummary import summary
import torch.nn.functional as F

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

        print(summary(model, (3, 224, 224)))
        
        #---- Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # normalize = transforms.Normalize([0.5, 0.5 ,0.5], [0.5, 0.5, 0.5])
        # transformList = []
        # transformList.append(transforms.Resize(transCrop))
        # transformList.append(transforms.ToTensor())
        # transformList.append(normalize)      
        
        # self.transformSequence = transforms.Compose(transformList)
    
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

        # normalize1 = transforms.Normalize([Rchan_mean,Gchan_mean,Bchan_mean], [Rchan_sd,Gchan_sd,Bchan_sd])
        normalize2 = transforms.Normalize([0.5, 0.5 ,0.5], [0.5, 0.5, 0.5])
        transformList = []
        transformList.append(transforms.Resize(224))
        transformList.append(transforms.ToTensor())
        # transformList.append(normalize1)
        transformList.append(normalize2)      
        self.transformSequence = transforms.Compose(transformList)


        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)

        input = torch.autograd.Variable(imageData)
        self.model.cuda()
        output = self.model(input.cuda())

        _, preds = torch.max(output, 1)
        print(preds)
        
        # outMean = output.view(224, 224, -1).mean(1)
        # print(outMean.data)
        # print(outMean)
        # outPRED = torch.cat((outPRED, outMean.data), 0)

        
        
        #---- Generate heatmap
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
        threshold = 0.5

        for i,eachList in enumerate(cam) :
            for j,each in enumerate(list(eachList)) :
                if each < threshold : cam[i][j] = None
                else : cam[i][j] = (cam[i][j]-threshold)/(1-threshold)
        
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        img = heatmap * 0.4 + imgOriginal
        cv2.imwrite(pathOutputFile, img)

def getImageData(pathDataFile) :
    labelList   = [ 'Normal','Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
                    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia' ]

    wholeDataList = list()
    for _ in range(15) :
        wholeDataList.append([])

    with open(pathDataFile,"r") as readStream :
        wholeData = readStream.readlines()
        for eachLine in wholeData :
            sepEachLine = eachLine.split()
            if sepEachLine[1:].count('1') == 0 :
                wholeDataList[0].append(sepEachLine[0])
            elif sepEachLine[1:].count('1') == 1 :
                wholeDataList[sepEachLine[1:].index('1')+1].append(sepEachLine[0])
            else :
                continue
    return (labelList,wholeDataList)

            


        
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
    nnClassCount    = 5
    transCrop       = 224
    pathModel       = os.path.join(".","models","categorical_4_-1to1Norm.pth.tar")
    # pathModel       = "m-12122019-143952.pth.tar"
    heatmapGen      = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
    print("Generator Loaded.")

    # pathInputImage = os.path.join("test","00009285_000.png")
    # pathOutputImage = os.path.join("test","heatmap_threshold_0.8.png")

    # pathDirData = '/home/memoming/study/CheXNet/database'
    pathDirData = '/srv/repo/users/memoming/CheXNet/database'

    caseNum = 4
    labelList, imageList = getImageData(os.path.join("dataIndex","test_1.txt"))

    with open(os.path.join("test","temp","index.txt"),"w") as indexStream :
        for i in range(len(labelList)) :
            eachLabel   = labelList[i]
            imgList     = random.sample(imageList[i],4)
            for no,eachPath in enumerate(imgList) :
                indexStream.write("heatmap_"+eachLabel+"_"+str(no)+".png"+"\t origin : "+eachPath+"\n")
                pathOutputImage = os.path.join("test","temp","heatmap_"+eachLabel+"_"+str(no)+".png")
                eachPath = os.path.join(pathDirData,eachPath)
                heatmapGen.generate(eachPath, pathOutputImage, transCrop)
    print("Done !")


    # for i,eachImagePath in enumerate(listImagePaths) :
    #     pathSaveDir     = os.path.join("/mnt","Hdd1","memoming","cheXNet","analyze","test_1_to_heatmap")
    #     pathInputImage  = eachImagePath
    #     pathOutputImage = os.path.join(pathSaveDir,eachImagePath.split(os.sep)[-1])
    #     heatmapGen.generate(pathInputImage, pathOutputImage, transCrop)
    #     print("\r","Generate ==> ",eachImagePath.split(os.sep)[-1],"(",i,"/",len(listImagePaths),")",end="")

