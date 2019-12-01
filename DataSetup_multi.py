import os
import subprocess
import multiprocessing
import platform


def unzipTask(eachZipFile) :
    pathDownloadDir = "C:\\Users\\memoming\\study\\CheXNet\\images.tar"
    pathDatabase    = "C:\\Users\\memoming\\study\\CheXNet\\database"
    eachUnzipPath   = os.path.join(pathDatabase,str(int(eachZipFile.split(".")[0].split("_")[-1])).zfill(3))
    if not os.path.exists(eachUnzipPath) : os.makedirs(eachUnzipPath)
    eachDirName     = "images_"+str(int(eachZipFile.split(".")[0].split("_")[-1])).zfill(3)
    subprocess.call("tar -zxvf "+os.path.join(pathDownloadDir,eachZipFile)+" -C "+eachUnzipPath, shell=True)
    if platform.system().lower() == "windows" :
        subprocess.call("move "+os.path.join(eachUnzipPath,"images")+" "+os.path.join(pathDatabase,eachDirName), shell=True)
    else :
        subprocess.call("mv "+os.path.join(eachUnzipPath,"images")+" "+os.path.join(pathDatabase,eachDirName), shell=True)


def main() :
    pathDownloadDir = "C:\\Users\\memoming\\study\\CheXNet\\images.tar"
    pathDatabase    = "C:\\Users\\memoming\\study\\CheXNet\\database"
    zipfileList     = os.listdir(pathDownloadDir)
    if not os.path.exists(pathDatabase) : os.makedirs(pathDatabase)

    numOfProcess = len(zipfileList)
    pool = multiprocessing.Pool(processes=numOfProcess)
    pool.map(unzipTask,zipfileList)


    # procs = []
    # for eachZip in zipfileList :
    #     p = multiprocessing.Process(target=unzipTask,args=(eachZip,))
    #     procs.append(p)
    #     p.start()

    # for p in procs :
    #     p.join()


if __name__ == "__main__" :
    main()



