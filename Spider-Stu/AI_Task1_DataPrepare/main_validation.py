import Step1_CreatePathTree as S1
import Step3_DownloadValidationData as S3

S1.CreatePathTree()
for sPath in S1.subPath:
    S3.DownloadPicBySubPath(sPath)
