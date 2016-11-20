
#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-
# 要使用中文就要加上面兩行

'''
Purpose:
    Load JSON data from file
'''

import commentjson as json

class Conf:
    def __init__(self, confPath):
        # load and store the configuration and update the object's dictionary
        #  confPath 所指向的檔案是儲存成 JSON 的檔案
        # 從 confPath 檔案裡產生 conf
        conf = json.loads(open(confPath).read())
        # conf = {"image_dataset": "datasets/caltech101/101_ObjectCategories/car_side"}
        # 將 conf 存到 __dict__
        self.__dict__.update(conf)

    def __getitem__(self, k):
        # python 的 dictionary.get(key, default)
        # 取 key 的 value, 若無此key 則回傳 default, default 預設值為 None
        return self.__dict__.get(k, None)


# class 都會有 __dict__ 來儲存該 class 的屬性
# vars() 會取出 __dict__ 所有的 key/value
