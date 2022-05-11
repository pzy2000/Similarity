from rest_framework.response import Response
from similarity2.database import Database
from similarity2.process import match_str2matrix, vector_match
from similarity2.cache import Cache
from typing import *
import torch

# 业务类型
BUSINESS_TYPE = "resource_resource"
# 数据库信息
db_data: Tuple[Tuple[Any]] = None
# 数据库match_str
db_match_str: List[str] = None
# 数据库词向量 (n_items, n_samples, n_features)
db_matrix: torch.Tensor = None
# 缓存对象
cache = Cache()


def init_model_vector(request):
    """
    :param business_type: 业务类型
    :return:

    初始化数据，初始化词向量

    """
    global db_data
    global db_match_str
    global db_matrix

    # 获取数据库交互对象
    db = Database.get_common_database()
    # 读取数据库信息
    db_data = db.filter(business_type=BUSINESS_TYPE).values_list("match_str", "original_Code", "original_data")
    db_match_str = db.filter(business_type=BUSINESS_TYPE).values_list("match_str", flat=True)
    # 计算数据库词向量
    if len(db_data) != 0:
        db_matrix = match_str2matrix(db_match_str)
    return Response(dict(code=200, data="", msg="初始化成功"))


def multiple_match(request):
    parameter = request.data
    # 读取请求参数
    request_data = parameter['data']
    k = parameter['k']
    percent = parameter['percent']

    # 处理请求参数k
    k = len(db_data) if k > len(db_data) else k

    # 处理请求参数percent
    if len(percent.split(',')) != 5:
        return Response({"code": 404, "msg": "权重配置错误！", "data": ''})
    percent = [float(x) for x in percent.split(',')]

    if len(db_data) == 0:
        return Response({"code": 404, "msg": "数据为空！", "data": ''})

    response_data = []
    for rd in request_data:
        match_str = rd['matchStr']
        request_id = rd['id']

        # 查看缓存
        if f"{match_str}{str(percent)}" in cache:
            # 从缓存中读取数据添加至结果
            response_data.append({"key": request_id, "result": cache.get(match_str)})
            continue

        # 处理请求match_str
        # (n_items,1,n_features)
        request_data_matrix = match_str2matrix(match_str)

        # 词向量匹配
        index, value = vector_match(X=db_matrix,
                                    y=request_data_matrix,
                                    weight=percent,
                                    k=k)
        result = [
            {
                "str": db_data[i][0],
                "originalCode": db_data[i][1],
                "originalData": db_data[i][2],
                "similarity": v
            }
            for i, v in zip(index, value)
        ]
        res = {
            "key": request_id,
            "result": result,
        }
        # 写入Cache
        cache.put(f"{match_str}{str(percent)}", result)
        response_data.append(res)

    return Response({
        "code": 200,
        "msg": "查询成功！",
        "data": response_data})
