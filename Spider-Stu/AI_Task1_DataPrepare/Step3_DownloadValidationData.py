import THKConfig
import requests
from bs4 import BeautifulSoup

import Tools

path_dict = {
    'dog': 'VW1fTcJewa9otu/UwruUxMVG5SMFJkJsZHSCqJHKhG9NsTOcvelJn5cNgoIocYb7j3YuTdSLaRdd7fYH4y0qNA==',
    'pig': 'VW1fTcJewa9otu/UwruUxBhG7vpkjgqCxtocKnTxRb1lspzb9NA0lvjqCdShz6+s',
    'bird': 'VW1fTcJewa9otu/UwruUxJ6scvPpJ/N5daTO0VRETHgRtX4egu3JnrlEbzPpC5QD',
    'girl': 'VW1fTcJewa9otu/UwruUxKGaBLe9/P3obo9KVnovlJQK5E7V788SlJTHqxGgVAVD',
    'snake': 'VW1fTcJewa9otu/UwruUxGnYsSqaqp/jtXg37xYcvNKcjvrQRMiCKWYKeNPOKI+2'
}

splash_url = 'http://192.168.146.128:8050/render.html'


def DownloadPicBySubPath(subPath):
    url = THKConfig.decrypt(path_dict.get(subPath))
    # Splash渲染的参数
    params = {
        'url': url,
        'wait': 5,  # 可以等待一定时间，确保页面加载完成
    }
    # 发起Splash请求
    response = requests.get(splash_url, params=params)
    # 使用Beautiful Soup解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    # 选取class="lazy"的img标签
    lazy_images = soup.find_all('img', class_='lazy')

    for img in lazy_images:
        previewUrl = img.get('data-original')[2:]
        print(img.get('alt'), "-->", previewUrl)
        if previewUrl.startswith('preview'):
            response = requests.get("https://" + previewUrl)
            if response.status_code == 200:
                fileID = Tools.genNewID()
                imageFile = Tools.getNewPath(subPath, fileID)
                with open(imageFile, 'wb') as file:
                    file.write(response.content)
