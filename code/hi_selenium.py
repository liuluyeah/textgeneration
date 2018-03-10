# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
import re


def find_sec():
    driver = webdriver.Chrome('D:\Applications\chromedriver_win32\chromedriver.exe')  # Get local session of firefox
    # browser.get("https://www.huobi.pro/zh-cn/eos_usdt/exchange/")  # Load page
    time.sleep(1)  # Let the page load
    while True:
        driver.get("https://www.huobi.pro/zh-cn/eos_usdt/exchange/")
        print(driver.title.split(' ')[0])



# 打印分区5下面的所有板块的当前在线人数列表
if __name__ == '__main__':
    # find_sec()
    driver = webdriver.PhantomJS(executable_path=r'E:\huobi\phantomjs.exe')
    while True:
        driver.get("https://www.huobi.pro/zh-cn/eos_usdt/exchange/")
        print(driver.title.split(' ')[0])
