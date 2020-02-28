# -*- coding: utf-8 -*-
 
# Importing Scrapy Library
import scrapy
from flipkart_reviews_scraping2.items import ReviewItem
 
# Creating a new class to implement Spide
class FlipkartReviewsSpider2(scrapy.Spider):
     
    # Spider name
    name = 'flipkartreviewsspider2'
    
    def start_requests(self):
        
        allowed_domains = ['flipkart.com']
        myBaseUrl = "https://www.flipkart.com/apple-iphone-7-black-32-gb/product-reviews/itmen6daftcqwzeg?pid=MOBEMK62PN2HU7EE&lid=LSTMOBEMK62PN2HU7EEINTGNU&marketplace=FLIPKART&page="
        urls=[]
        for i in range(1,826):
            urls.append(myBaseUrl+str(i))
            for url in urls:
                yield scrapy.Request(url=url, callback=self.parse)    
                
                
    def parse(self, response):
        temp=response.xpath('//div[@class="qwjRop"]//div//div').getall()
        tem=len(temp)
        print(tem)

        for i in range(0,tem):
            item=ReviewItem()
            item['star_rating']=response.xpath('//div[@class="hGSR34 E_uFuv"]/text()')[i].get()
            comments1=response.xpath('//div[@class="qwjRop"]//div//div')[i].get()
            item['comments']=comments1.replace('<br>',' ').replace('<div class="">',' ').replace('</div>','')  
            yield item
