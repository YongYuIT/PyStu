import os.path
import shutil
import base64
import requests

import FileToString as fileinput
from bs4 import BeautifulSoup

# 从using.html文件中读取html内容
html_content = fileinput.readFromFile('using.html')
# 使用Beautiful Soup解析HTML
soup = BeautifulSoup(html_content, 'html.parser')

print(
    "===============================================================================================================================")
# 提取满足条件的<div>标签
# 1、id以emb打头
# 2、style=display:none
target_divs = soup.find_all(
    lambda tag: tag.name == 'div' and tag.get('id', '').startswith('emb') and 'display:none' in tag.get('style', ''))

print("get divs:", len(target_divs))

pic_path = "tmp_pic"
if os.path.exists(pic_path):
    shutil.rmtree(pic_path)

os.makedirs(pic_path)

for div in target_divs:
    pic_base64 = div.text.split(",")[1];
    print(pic_base64)
    # 解码Base64字符串为字节数据
    image_data = base64.b64decode(pic_base64)
    # 写入解码后的图片数据到文件
    imageFile = pic_path + "/" + div.get('id', '') + ".png"
    with open(imageFile, 'wb') as file:
        file.write(image_data)

print(
    "===============================================================================================================================")
# 提取满足条件的img
target_imgs = soup.find_all(
    lambda tag: tag.name == 'img' and 'mimg' in tag.get('class', ''))
print("get imgs:", len(target_imgs))

index = 0
for img in target_imgs:
    print(img)
    image_url = img.get("src", "")
    if image_url.startswith("http"):
        # 发送HTTP请求获取图片数据
        response = requests.get(image_url)
        # 检查响应状态码，200表示成功
        if response.status_code == 200:
            # 以二进制写入模式打开文件，将图片内容写入文件
            imageFile = pic_path + "/" + str(index) + ".png"
            with open(imageFile, 'wb') as file:
                file.write(response.content)
            index += 1
    else:
        pic_base64 = image_url.split(",")[1];
        # 解码Base64字符串为字节数据
        image_data = base64.b64decode(pic_base64)
        # 写入解码后的图片数据到文件
        imageFile = pic_path + "/" + str(index) + ".png"
        with open(imageFile, 'wb') as file:
            file.write(image_data)
        index += 1
