import requests

# 要爬取的目标网页URL
url = 'https://image.baidu.com/search/index?tn=baiduimage&word=dog'

# 发起Splash请求
response = requests.get(url)

# 打印渲染后的页面内容
print(response.text)
