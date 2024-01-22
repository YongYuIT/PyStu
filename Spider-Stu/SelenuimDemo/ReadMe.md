# selenium 开发环境搭建

1. 安装python selenium依赖库

参照：https://www.selenium.dev/zh-cn/documentation/webdriver/getting_started/install_library/

为了安装最新版本，建议下载源码安装

下载：https://files.pythonhosted.org/packages/16/fd/a0ef793383077dd6296e4637acc82d1e9893e9a49a47f56e96996909e427/selenium-4.16.0.tar.gz

得到selenium-4.16.0.tar.gz

* 使用Anaconda准备一个空白的python环境，本例中python版本选3.11.7
* 从Anaconda中这个空白环境启动一个Terminal（注意不是Open With Python）
* 可以看到这个Terminal工作目录是“C:\user\当前用户”
* 将解压后的selenium-4.16.0目录拷贝到这个工作目录下
* cd到selenium-4.16.0目录，运行python setup.py install

看到如下输出，则表示安装成功

~~~
Installed 你的python环境目录\lib\site-packages\pycparser-2.21-py3.11.egg
Finished processing dependencies for selenium==4.16.0
~~~

2. 安装ChromeDriver

由于selenium的工作原理是：

你的python程序 --调用--> selenium库 --调用--> ChromeDriver（或其他的浏览器驱动） --调用--> 本地浏览器

所以需要安装这个ChromeDriver

由于在selenium在4.6版本以后，支持由Selenium manages自动下载ChromeDriver，所以无需自行查找并下载浏览器对应的ChromeDriver

关于这问题，官网原为是这样说的

~~~
As of Selenium 4.6, Selenium downloads the correct driver for you. You shouldn’t need to do anything.
~~~

如果想借助Selenium manages自动安装的话，需要先安装ChromeDriverManager库（用conda和pip都可以）

~~~
pip install webdriver-manager
~~~

在代码中这样使用

~~~
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
~~~

但是如果想手动安装的话，也可以访问

https://chromedriver.chromium.org/downloads

找到自己chrome版本对应的ChromeDriver，例如我的chrome版本是

![chrome-version.png](ReadMePic%2Fchrome-version.png)

能找到最接近的版本是

https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/120.0.6099.109/win64/chromedriver-win64.zip

下载完成之后，解压执行

~~~
.\chromedriver.exe --version
ChromeDriver 120.0.6099.109 (3419140ab665596f21b385ce136419fde0924272-refs/branch-heads/6099@{#1483})
~~~

可以将这个chromedriver.exe所在目录加入到PATH里面去，也可以直接通过代码加载这个驱动

~~~
from selenium.webdriver.chrome.options import Options
opt = Options()
opt.binary_location = './chromedriver-win64/chromedriver.exe'
driver = webdriver.Chrome(options=opt)
~~~

* 特别声明，上面借助Selenium manages自动安装是跑通了的；手动安装没有跑通

3. 参照 https://www.selenium.dev/documentation/webdriver/getting_started/first_script/ 重写 FirstDemo

全文在：https://github.com/SeleniumHQ/seleniumhq.github.io/blob/trunk/examples/python/tests/getting_started/first_script.py#L6


