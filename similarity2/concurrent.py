"""
多线程有关代码
"""
from concurrent import futures


"""
没明白similarity中RejectQueue的用途是什么。
当qsize >= maxsize 时，任务会被丢弃（有什么用？）猜测：查询线程超过20个即丢弃？防止服务器崩溃or响应过慢？
这里先采用ThreadPoolExecutor代替
"""
EXECUTOR = futures.ThreadPoolExecutor(max_workers=5)
