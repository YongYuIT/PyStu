import requests
import StringToFile as fpring

# Splash服务的地址
splash_url = 'http://192.168.146.128:8050/render.html'

# 要爬取的目标网页URL
url = 'https://www.bing.com/images/search?q=dog&qs=n&form=QBILPG&sp=-1&lq=0&pq=d&sc=10-1&cvid=1945A488FD6E4594ABA53A0E422C90D0&ghsh=0&ghacc=0&first=1'

# Splash请求参数，指定要渲染的网页URL和一些其他选项
params = {
    'url': url,
    'wait': 10,  # 等待页面加载的时间（单位：秒）
    'render_all': 1,  # 加载所有JavaScript代码
    'images': 1,  # 加载所有图片
}

# 发起Splash请求
response = requests.get(splash_url, params=params)

# 打印渲染后的页面内容
fpring.printToFile(response.text, "using.html")
