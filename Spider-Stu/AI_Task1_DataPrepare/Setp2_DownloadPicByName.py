import Step1_CreatePathTree as S1
import requests
from bs4 import BeautifulSoup
import uuid
import base64

searchNames = ['小狗', '猪', '小鸟', '美女', '毒蛇']
nameDic = dict(zip(S1.subPath, searchNames))


def GetFullPath(name):
    return S1.rootPath + "/" + name


splash_url = 'http://192.168.146.128:8050/execute'

base_url = 'https://cn.bing.com/images/search?q=####&first=1'

lua_script = """
function main(splash)
    local num_scrolls = 5
    local scroll_delay = 2.0

    splash:go(splash.args.url)
    splash:wait(2)

    for _ = 1, num_scrolls do
        splash:runjs("window.scrollTo(0, document.body.scrollHeight);")
        splash:wait(scroll_delay)
    end

    return splash:html()
end
"""

params = {
    "wait": 0.5,
    "resource_timeout": 0,
    "viewport": "1024x768",
    "render_all": 0,
    "images": 1,
    "http_method": "GET",
    "html5_media": 0,
    "http2": 0,
    "save_args": [],
    "load_args": {},
    "timeout": 90,
    "request_body": 0,
    "response_body": 0,
    "engine": "webkit",
    "har": 1,
    "png": 1,
    "html": 1,
    'lua_source': lua_script,
}


def genNewID():
    random_uuid = uuid.uuid4()
    return str(random_uuid)


def getNewPath(subPath, name):
    return S1.rootPath + "/" + subPath + "/" + name + ".png"


def DownloadPicBySubPath(subPath):
    searchName = nameDic[subPath]
    url = base_url.replace("####", searchName)
    params['url'] = url
    pageResponse = requests.get(splash_url, params=params)
    html_content = pageResponse.text
    soup = BeautifulSoup(html_content, 'html.parser')
    target_imgs = soup.find_all(
        lambda tag: tag.name == 'img' and 'mimg' in tag.get('class', ''))
    all_ids = ""
    for img in target_imgs:
        image_url = img.get("src", "")
        if image_url.startswith("http"):
            response = requests.get(image_url)
            if response.status_code == 200:
                fileID = genNewID()
                imageFile = getNewPath(subPath, fileID)
                with open(imageFile, 'wb') as file:
                    file.write(response.content)
                all_ids += fileID + '\n'
        else:
            pic_base64 = image_url.split(",")[1];
            image_data = base64.b64decode(pic_base64)
            fileID = genNewID()
            imageFile = getNewPath(subPath, fileID)
            with open(imageFile, 'wb') as file:
                file.write(image_data)
            all_ids += fileID + '\n'
    idsFile = S1.rootPath + "/" + subPath + "/ids.txt"
    with open(idsFile, 'w', encoding='utf-8') as file:
        file.write(all_ids)
