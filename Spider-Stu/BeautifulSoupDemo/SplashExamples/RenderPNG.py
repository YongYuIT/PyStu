import requests
import StringToFile as fprint
import THKConfig
from bs4 import BeautifulSoup

# Splash服务的地址，render用于加载动态网页
splash_url = 'http://192.168.146.128:8050/render.html'

# 要爬取的目标网页URL
url = THKConfig.decrypt('VW1fTcJewa9otu/UwruUxMVG5SMFJkJsZHSCqJHKhG9NsTOcvelJn5cNgoIocYb7j3YuTdSLaRdd7fYH4y0qNA==')

# Splash渲染的参数
params = {
    'url': url,
    'wait': 5,  # 可以等待一定时间，确保页面加载完成
}

# 发起Splash请求
response = requests.get(splash_url, params=params)

# 打印渲染后的页面内容
fprint.printToFile(response.text, "using.html")

# 使用Beautiful Soup解析HTML
soup = BeautifulSoup(response.text, 'html.parser')
# 选取class="lazy"的img标签
lazy_images = soup.find_all('img', class_='lazy')

for img in lazy_images:
    print(img.get('alt'), "-->", img.get('data-original'))
