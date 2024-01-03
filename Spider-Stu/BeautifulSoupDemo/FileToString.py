import os


def readFromFile(fileName):
    if not os.path.exists(fileName):
        return ''
    with open(fileName, 'r', encoding='utf-8') as file:
        file_content = file.read()
        return file_content
