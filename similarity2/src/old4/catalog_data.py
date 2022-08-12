from rest_framework.response import Response

from demo.settings import DEBUG
from similarity2.database import Database
from similarity2.process import match_str2matrix, vector_match
from similarity2.cache import Cache
from typing import *
import torch

# 业务类型
BUSINESS_TYPE = "catalog_data"
# 数据库信息
db_data: List[Tuple[Any, Any, Any]] = None
# 数据库match_str
db_match_str: List[str] = None
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
    if DEBUG:
        print(f"初始化:\nbusiness_type: {BUSINESS_TYPE}\n数据库信息：{db_data}\n\n")
    # 计算数据库词向量
    if len(db_data) != 0:
        db_matrix = match_str2matrix(db_match_str)
    # 清除缓存
    cache.clear()
    return Response(dict(code=200, data="", msg="初始化成功"))


def __get_filter_data(department_id: str, style: int):
    """
    获取经过筛选后的db_data,db_matrix,db_match_str
    catalog_data这个推荐接口加两个入参
    departmentId  部门id
    style 方式
    当style为1时   只推荐本部门的数据
    当style为2时   只推荐其他部门的数据
    """
    filter_db_data = db_data
    filter_db_matrix = db_matrix
    filter_db_match_str = db_match_str
    import json
    if department_id and style:
        style = int(style)
        assert (style == 1 or style == 2), "参数style必须为1或2"

        index = [i for i, v in enumerate(db_data) if
                 (json.loads(v[2])['departmentId'] == department_id) ^ (style - 1) # 使用style异或即可
                 ]
        if len(index) == 0:
            raise ValueError(f"departmentId:{department_id},数据为空！")

        filter_db_data = [db_data[i] for i in index]
        filter_db_matrix = db_matrix[:, index, :]
        filter_db_match_str = [db_match_str[i] for i in index]

    return filter_db_data, filter_db_matrix, filter_db_match_str


def multiple_match(request):
    parameter = request.data
    # 读取请求参数
    request_data = parameter['data']
    k = parameter['k']
    percent = parameter['percent']
    department_id = request.data.get('departmentId', '')
    style = request.data.get('style', '')

    # department_id && style 相关筛选：
    if department_id and not style:
        return Response({"code": 400, "msg": "未提供style参数", "data": ''})
    try:
        db_data, db_matrix, db_match_str = __get_filter_data(department_id=department_id, style=style)
    except Exception as e:
        return Response({"code": 404, "msg": str(e), "data": ''})
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
        if f"{match_str}{str(percent)}{k}{department_id}{style}" in cache:
            # 从缓存中读取数据添加至结果
            response_data.append(
                {"key": request_id, "result": cache.get(f"{match_str}{str(percent)}{k}{department_id}{style}")})
            continue

        request_data_matrix = match_str2matrix(match_str)

        # 词向量匹配
        index, value, items_value = vector_match(X=db_matrix,
                                                 y=request_data_matrix,
                                                 weight=percent,
                                                 k=k)
        result = [
            {
                "str": db_data[i][0],
                "originalCode": db_data[i][1],
                "originalData": db_data[i][2],
                "similarity": v,
                "items_similarity": item_v,
            }
            for i, v, item_v in zip(index, value, items_value)
        ]
        res = {
            "key": request_id,
            "result": result,
        }
        # 写入Cache
        cache.put(f"{match_str}{str(percent)}{k}{department_id}{style}", result)
        response_data.append(res)

    return Response({
        "code": 200,
        "msg": "查询成功！",
        "data": response_data})


def increment_data(request):
    global db_data
    global db_match_str
    global db_matrix
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        if len(match_str.split('^')) != 5:
            return Response({"code": 200, "msg": "新增数据失败，有效数据字段不等于5", "data": ""})

        # 加入数据集
        db_data.append((match_str, original_code, original_data))
        # 加入match_str集合
        db_match_str.append(match_str)
        # 计算词向量
        vec = match_str2matrix(match_str)
        # 加入词向量集合
        db_matrix = torch.cat((db_matrix, vec), dim=1)
    # 清除缓存
    cache.clear()
    return Response({"code": 200, "msg": "新增数据成功！", "data": ""})


def delete_data(request):
    global db_data
    global db_match_str
    global db_matrix
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        data = (match_str, original_code, original_data)

        # 时间复杂度O(n)，字典啥的再说
        try:
            index = db_data.index(data)
        except ValueError as e:
            return Response({"code": 200, "msg": "无该数据！", "data": ""})

        # 删除数据
        del db_data[index]
        del db_match_str[index]
        db_matrix = db_matrix[:, torch.arange(db_matrix.size(1)) != index, :]
        # 清除缓存
        cache.clear()
        return Response({"code": 200, "msg": "删除数据成功！", "data": ""})
