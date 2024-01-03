import os


def printToFile(strContent, fileName):
    # 检查文件是否存在，如果存在则删除
    if os.path.exists(fileName):
        os.remove(fileName)
    # 以写入模式打开文件，如果文件不存在则创建，如果存在则覆盖内容
    with open(fileName, 'w', encoding='utf-8') as file:
        file.write(strContent)
