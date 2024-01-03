# 安装 splash

## 步骤1：ubuntu更新国内源

图形界面操作

设置(Settings)-->关于(About)-->软件和更新(Sofeware Updates)-->Ubuntu软件(Ubuntu software)
下载自(Download from)

默认是：Server for United States
改成：中国，mirrors.aliyun.com

执行

~~~shell
sudo apt-get update
sudo apt-get upgrade
~~~

## 步骤2：安装docker

参照: https://docs.docker.com/engine/install/ubuntu/#install-from-a-package

查看Ubuntu系统版本号

~~~shell
cat /etc/os-release
~~~

取UBUNTU_CODENAME字段，到

https://download.docker.com/linux/ubuntu/dists 

找到Ubuntu版本号对应的子目录

https://download.docker.com/linux/ubuntu/dists/focal

在该目录下找到pool/stable/下合适CPU架构的deb package

https://download.docker.com/linux/ubuntu/dists/focal/pool/stable/amd64/

下载同一个版本号的

containerd.io_<version>_<arch>.deb
docker-ce_<version>_<arch>.deb
docker-ce-cli_<version>_<arch>.deb
docker-buildx-plugin_<version>_<arch>.deb
docker-compose-plugin_<version>_<arch>.deb

确保

~~~
ls
containerd.io_1.6.26-1_amd64.deb                            docker-ce-cli_24.0.7-1~ubuntu.20.04~focal_amd64.deb
docker-buildx-plugin_0.11.2-1~ubuntu.20.04~focal_amd64.deb  docker-compose-plugin_2.21.0-1~ubuntu.20.04~focal_amd64.deb
docker-ce_24.0.7-1~ubuntu.20.04~focal_amd64.deb
~~~

执行

~~~
sudo dpkg -i containerd.io_1.6.26-1_amd64.deb
sudo dpkg -i docker-ce-cli_24.0.7-1~ubuntu.20.04~focal_amd64.deb
sudo dpkg -i docker-ce_24.0.7-1~ubuntu.20.04~focal_amd64.deb
sudo dpkg -i docker-buildx-plugin_0.11.2-1~ubuntu.20.04~focal_amd64.deb
sudo dpkg -i docker-compose-plugin_2.21.0-1~ubuntu.20.04~focal_amd64.deb
~~~

验证

~~~
sudo service docker start
sudo docker run hello-world
~~~

## 步骤3：安装splash

修改docker配置，新增国内docker源

~~~
sudo gedit /etc/docker/daemon.json
~~~

registry-mirrors字段新增

https://registry.docker-cn.com

~~~
{
  "registry-mirrors": [
    "https://registry.docker-cn.com"
  ]
}
~~~

重启docker

~~~
systemctl restart docker
~~~

参照: https://splash.readthedocs.io/en/stable/install.html#linux-docker

~~~
sudo docker pull scrapinghub/splash
sudo docker run -it -p 8050:8050 --rm scrapinghub/splash
~~~

