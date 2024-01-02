import scrapy


class DogImagesSpider(scrapy.Spider):
    name = 'dog_images'
    start_urls = [
        'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&fm=index&pos=history&word=%E5%B0%8F%E7%8B%97']
    basic_common_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, headers=self.basic_common_headers, callback=self.parse)

    def parse(self, response):
        print("info-->at parse-->", response)
        # image_links = response.css('img.main_img.img-hover::attr(src)').extract()
        image_links = response.xpath('//img[contains(@class, "main_img") and contains(@class, "img-hover")]/@src')
        print("parse size-->", len(image_links))
        for link in image_links:
            yield scrapy.Request(url=link, headers=self.basic_common_headers, callback=self.save_image)

    def save_image(self, response):
        print("info-->at save_image-->", response)
        image_url = response.url
        image_name = image_url.split('/')[-1]
        with open(image_name, 'wb') as f:
            f.write(response.body)
