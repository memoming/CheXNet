import os
import subprocess

#pathDownloadDir = "/srv/repo/users/memoming/CheXNet/wholeData"
# pathDownloadDir = "/mnt/NAS_Mount1/Janghoon_temp/20191112_OpenDataSets/NIH_Chest_Data/RawData"
pathDownloadDir = "C:\\Users\\memoming\\study\\CheXNet\\images.tar"
#pathDatabase    = "/srv/repo/users/memoming/CheXNet/database"
pathDatabase    = "C:\\Users\\memoming\\study\\CheXNet\\database"
zipfileList     = os.listdir(pathDownloadDir)

if not os.path.exists(pathDatabase) : os.makedirs(pathDatabase)

for eachZipFile in zipfileList :
    eachDirName = "images_"+str(int(eachZipFile.split(".")[0].split("_")[-1])).zfill(3)
    subprocess.call("tar -zxvf "+os.path.join(pathDownloadDir,eachZipFile)+" -C "+pathDatabase, shell=True)
    subprocess.call("mv "+os.path.join(pathDatabase,"images")+" "+os.path.join(pathDatabase,eachDirName), shell=True)

print("Data Move to ...\t",pathDatabase,"\n\n")  



