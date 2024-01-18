from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import THKConfig
import time

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
driver.get(
    THKConfig.decrypt('RYmHaHDMRKF71PvKY4W8uJgi/wg1Ve67GQvHdvQjvWBFbmeengrnTYW2UGYRk3pMqdwgq9ZLWt1u/XOZ6PeaNw=='))

# 等待一段时间，确保页面加载完毕
time.sleep(4)

# 设置下拉次数和间隔时间
total_pulls = 10
pull_interval = 4

# 执行自动下拉
for _ in range(total_pulls):
    # 使用JavaScript执行滚动操作
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # 等待一段时间，确保内容加载完毕
    time.sleep(pull_interval)

# 获取网页的HTML内容
html_content = driver.page_source

# 关闭浏览器
driver.quit()

# 使用Beautiful Soup解析HTML
soup = BeautifulSoup(html_content, 'html.parser')
# 找到所有class为'main_img'的img标签
img_tags = soup.find_all('img', class_='main_img')
# 遍历所有匹配的标签并获取data-imgurl属性的值
for img_tag in img_tags:
    data_imgurl = img_tag['data-imgurl']
    print(data_imgurl)
