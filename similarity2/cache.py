"""系统缓存相关代码"""
from collections import OrderedDict


class Cache:
    """通过有序字典实现FIFO缓存"""

    def __init__(self, maxsize=10000):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def clear(self):
        self.cache.clear()

    def put(self, key, value):
        if len(self.cache) + 1 > self.maxsize:
            # FIFO 顺序弹出
            self.cache.popitem(last=False)
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key, None)

    def size(self):
        return len(self.cache)

    def __contains__(self, item):
        return item in self.cache
