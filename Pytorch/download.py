#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/17 9:52
#@Author: czj
#@File  : download.py

# import pandas as pd
# import numpy as np
# import requests
# import numpy as np
# import os
# import time
# if __name__ == "__main__":
# path = "./"
# for i in np.arange(262,300):
#     downloadAddress = "https://github.com/neheller/kits19/raw/master/data/case_00"+str(i)+"/imaging.nii.gz"
#     print("is Download:",str(i)+"/imaging.nii.gz")
#     savePath = path+str(i)
#     if not os.path.exists(savePath):
#         os.mkdir(savePath)
#     f = requests.get(downloadAddress)
#     with open(savePath+"/imaging.nii.gz", "wb") as code:
#         code.write(f.content)
#     time.sleep(10)
#
#
# i = 260
# downloadAddress = "https://github.com/neheller/kits19/raw/master/data/case_00"+str(i)+"/imaging.nii.gz"
# print("is Download:",str(i)+"/imaging.nii.gz")
# savePath = path+str(i)
# if not os.path.exists(savePath):
#     os.mkdir(savePath)
# f = requests.get(downloadAddress)
# a = requests.head(downloadAddress)
# with open(savePath+"/imaging.nii.gz", "wb") as code:
#     code.write(f.content)
# time.sleep(10)
import requests
import numpy as np
import os
import time
from tqdm import tqdm

# import random
# user_agent = [
# 	"Mozilla/5.0 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)",
# 	"Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
# 	"Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
# 	"Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
# 	"Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
# 	"Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
# 	"Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
# 	"Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
# 	"Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
# 	"Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
# 	"Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
# 	"Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
# 	"Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
# 	"Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
# 	"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
# 	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
# 	"Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52"
# ]
#
# headers = {
# 'User-Agent': random.choice(user_agent),  # 浏览器头部
# 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', # 客户端能够接收的内容类型
# 'Accept-Language': 'en-US,en;q=0.5', # 浏览器可接受的语言
# 'Connection': 'keep-alive', # 表示是否需要持久连接
# }
#
#
def downfile(url, filename):
    # r = requests.get(url, stream=True,headers=headers)
    r = requests.get(url, stream=True, headers=headers)
    with open(filename, "wb") as code:
        for chunk in tqdm(r.iter_content(chunk_size=1024)):
            if chunk:
                code.write(chunk)

if __name__=="__main__":
    for i in np.arange(262, 300):
        downloadAddress = "https://github.com/neheller/kits19/raw/master/data/case_00" + str(i) + "/imaging.nii.gz"
        path = './'
        savePath = path + str(i)
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        downfile(downloadAddress, savePath + "/imaging.nii.gz")
        time.sleep(10)

