from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import THKConfig
import time
import requests
import Tools

path_dict = {
    'dog': 'RYmHaHDMRKF71PvKY4W8uJgi/wg1Ve67GQvHdvQjvWBFbmeengrnTYW2UGYRk3pMqdwgq9ZLWt1u/XOZ6PeaNw==',
    'pig': 'RYmHaHDMRKF71PvKY4W8uJgi/wg1Ve67GQvHdvQjvWBFbmeengrnTYW2UGYRk3pM5ghLB2eWDfXLlA16U+aOzg==',
    'bird': 'RYmHaHDMRKF71PvKY4W8uJgi/wg1Ve67GQvHdvQjvWBFbmeengrnTYW2UGYRk3pM+WlULS6DDWhNpCxRgvpfgw==',
    'girl': 'RYmHaHDMRKF71PvKY4W8uJgi/wg1Ve67GQvHdvQjvWBFbmeengrnTYW2UGYRk3pM6TfjqXCPG0a7UgB3egYDtw==',
    'snake': 'RYmHaHDMRKF71PvKY4W8uJgi/wg1Ve67GQvHdvQjvWBFbmeengrnTYW2UGYRk3pMYK/4XXYYuAczuYnW+eh/DQ=='
}


def DownloadPicBySubPath(subPath):
    url = THKConfig.decrypt(path_dict.get(subPath))
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(url)
    # 等待一段时间，确保页面加载完毕
    time.sleep(4)
    # 设置下拉次数和间隔时间
    total_pulls = 100
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
        response = requests.get(data_imgurl)
        if response.status_code == 200:
            fileID = Tools.genNewID()
            imageFile = Tools.getNewPath(subPath, fileID)
            with open(imageFile, 'wb') as file:
                file.write(response.content)
