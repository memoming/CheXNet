import os
import subprocess

pathDownloadDir = "/srv/repo/users/memoming/CheXNet/wholeData"
pathDatabase    = "/srv/repo/users/memoming/CheXNet/database"
logFile         = open("DataSetupLog.txt","w",encoding="utf-8")
zipfileList     = os.listdir(pathDownloadDir)

if not os.path.exists(pathDatabase) : os.makedirs(pathDatabase)

for eachZipFile in zipfileList :
    print("Unzip ...\t", eachZipFile,end="")
    eachDirName = "images_"+str(int(eachZipFile.split(".")[0].split("_")[-1])).zfill(3)
    logFile.write("tar -zxvf "+os.path.join(pathDownloadDir,eachZipFile)+" -C "+pathDatabase+"\n\n")
    subprocess.call("tar -zxvf "+os.path.join(pathDownloadDir,eachZipFile)+" -C "+pathDatabase, \
                    stdin=None, stdout=logFile, stderr=logFile, shell=True)
    print(" ===> Done !")
    print("Move to ...\t",os.path.join(pathDatabase,eachDirName),"\n\n")
    logFile.write("mv "+os.path.join(pathDatabase,"images")+" "+os.path.join(pathDatabase,eachDirName)+"\n\n")           
    subprocess.call("mv "+os.path.join(pathDatabase,"images")+" "+os.path.join(pathDatabase,eachDirName), \
                    stdin=None, stdout=logFile, stderr=logFile, shell=True)
logFile.close()



