import Step1_CreatePathTree as S1
import Step4_DownloadMorePic as S4

S1.CreatePathTree()
for sPath in S1.subPath:
    S4.DownloadPicBySubPath(sPath)
