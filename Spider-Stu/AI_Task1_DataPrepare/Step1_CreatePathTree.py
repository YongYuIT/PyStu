import os
import shutil

rootPath = "pic"
subPath = ['dog', 'pig', 'bird', 'girl', 'snake']


def CreatePathTree():
    if os.path.exists(rootPath):
        shutil.rmtree(rootPath)
    os.makedirs(rootPath)
    for subPathName in subPath:
        currentSubPath = rootPath + "/" + subPathName
        os.makedirs(currentSubPath)

