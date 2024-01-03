import requests
import StringToFile as fprint

# Splash服务的地址，注意：如果带有自定义Lua逻辑，此处需要用execute
splash_url = 'http://192.168.146.128:8050/execute'

# 要爬取的目标网页URL
url = 'https://www.bing.com/images/search?q=%E7%8B%97%E7%8B%97%E5%85%A8%E8%BA%AB&qs=n&form=QBIR&sp=-1&lq=0&pq=%E7%8B%97%E7%8B%97%E5%85%A8%E8%BA%AB&sc=10-4&cvid=B6C616DB0EFE47FA877178F47EFE6BD6&ghsh=0&ghacc=0&first=1'

# Splash执行的JavaScript脚本，模拟页面滚动以加载更多图片
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

# 调试上述Lua代码，可浏览器打开http://192.168.146.128:8050
# 在RenderMe中填入目标url和需要调试的Lua脚本
# 可以从日志中拿到界面调试Lua脚本时所使用的Splash请求参数，可以借鉴


# Splash请求参数，指定要渲染的网页URL和一些其他选项
params = {
    'url': url,
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

# 发起Splash请求
response = requests.get(splash_url, params=params)

# 打印渲染后的页面内容
fprint.printToFile(response.text, "using.html")
