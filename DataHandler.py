import os
import itertools


pathOriginDB     = os.path.join("dataIndex","test_1.txt")
pathNewTrainDB  = os.path.join("dataIndex","test_categorical.txt")
wholeLine       = None
maskList        = ["L","C","P","L","L","L","L","P","L","L","L","L","P","H"]

with open(pathOriginDB, "r") as stream :
    wholeLine = stream.readlines()

with open(pathNewTrainDB,"w") as stream :
    for eachLine in wholeLine :
        eachRlt         = ""
        eachPath        = eachLine.split()[0]
        eachLabelList   = eachLine.split()[1:]
        eachNewLabelList = [0,0,0,0,0]

        eachRlt += eachPath + " "
        if not "1" in eachLabelList :
            eachNewLabelList[4] = 1
            for l in eachNewLabelList :
                eachRlt += str(l) + " "
        else :
            for i in range(len(eachLabelList)) :
                if eachLabelList[i] == "1" :
                    if maskList[i] == "L" :
                        eachNewLabelList[0] = 1
                    elif maskList[i] == "C" :
                        eachNewLabelList[1] = 1
                    elif maskList[i] == "P" :
                        eachNewLabelList[2] = 1
                    else :
                        eachNewLabelList[3] = 1
            for l in eachNewLabelList :
                eachRlt += str(l) + " "
        stream.write(eachRlt+"\n")
        print(eachRlt[:-1])