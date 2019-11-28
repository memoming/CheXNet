import os
import subprocess

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', \
                'Mass', 'Nodule', 'Pneumonia','Pneumothorax', \
                'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', \
                'Pleural_Thickening', 'Hernia']

pathWholeImageDir = "/mnt/Hdd1/memoming/cheXNet/analyze/test_1_to_heatmap"


pathDatasetFile    = os.path.join(".","dataIndex","test_1.txt")
pathImageDirectory = os.path.join(".","database")
listImagePaths     = []
listImageLabels    = []
fileDescriptor     = open(pathDatasetFile, "r")

line = True
while line:
    line = fileDescriptor.readline()
    if line:
        lineItems       = line.split()
        eachImageFile   = lineItems[0].split("/")[-1]
        imageLabel      = lineItems[1:]
        imageLabel      = [int(i) for i in imageLabel]
        targetDirName   = ""
        for i,flag in enumerate(imageLabel) :
            if flag == 1 :
                if targetDirName == "" :
                    targetDirName = CLASS_NAMES[i]
                else :
                    targetDirName = "-".join([targetDirName,CLASS_NAMES[i]])
        if targetDirName == "" : targetDirName = "Normal"
        
        pathTargetDir = None
        if not os.path.exists(os.path.join(pathWholeImageDir,targetDirName)) :
            os.makedirs(os.path.join(pathWholeImageDir,targetDirName))
        pathTargetDir = os.path.join(pathWholeImageDir,targetDirName)
        subprocess.call("mv "+\
                            os.path.join(pathWholeImageDir,eachImageFile)\
                            +" "+pathTargetDir,shell=True)

fileDescriptor.close()
