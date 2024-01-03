import requests
import StringToFile as fpring

# 要爬取的目标网页URL
url = 'https://www.bing.com/images/search?q=dog&qs=n&form=QBILPG&sp=-1&lq=0&pq=d&sc=10-1&cvid=1945A488FD6E4594ABA53A0E422C90D0&ghsh=0&ghacc=0&first=1'

# 发起Splash请求
response = requests.get(url)

# 打印渲染后的页面内容
fpring.printToFile(response.text, "without.html")
