import Step1_CreatePathTree as S1
import Setp2_DownloadPicByName as S2

S1.CreatePathTree()
for sPath in S1.subPath:
    S2.DownloadPicBySubPath(sPath)
