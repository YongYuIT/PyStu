import uuid
import Step1_CreatePathTree as S1

def genNewID():
    random_uuid = uuid.uuid4()
    return str(random_uuid)

def getNewPath(subPath, name):
    return S1.rootPath + "/" + subPath + "/" + name + ".png"