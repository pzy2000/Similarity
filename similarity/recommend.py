# coding=utf-8

from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

# =============similarity2===============
from similarity2.src.new4 import resource_resource, column_meta, column_terminology, resource_terminology
from similarity2.src.old4 import model_data, data_model, catalog_data, item_material

'''
数据推荐总入口，实现内容包括：
1. 需求一
2. 需求二
3. 需求三
4. 需求四
'''


# 启动项目时自动初始化
class FakeRequest:
    def __init__(self):
        self.data = None


fake_request = FakeRequest()
catalog_data.init_model_vector(fake_request)
item_material.init_model_vector(fake_request)
data_model.init_model_vector(fake_request)
model_data.init_model_vector(fake_request)
column_meta.init_model_vector(fake_request)
column_terminology.init_model_vector(fake_request)
resource_resource.init_model_vector(fake_request)
resource_terminology.init_model_vector(fake_request)


@csrf_exempt
@api_view(http_method_names=['post'])
@permission_classes((permissions.AllowAny,))
def multiple_match(request):
    parameter = request.data
    business_type = parameter['businessType']
    if business_type == 'catalog_data':
        # 需求一，目录数据推荐
        return catalog_data.multiple_match(request)
    elif business_type == 'item_material':
        # 需求二，给定事项材料关联目录
        return item_material.multiple_match(request)
    elif business_type == 'data_model':
        # 需求三，根据数据表字段推荐模型属性
        return data_model.multiple_match(request)
    elif business_type == 'model_data':
        # 需求四，根据模型表属性推荐数据字段
        return model_data.multiple_match(request)
    elif business_type == "column_meta":
        # 需求五 数据元推荐
        return column_meta.multiple_match(request)
    elif business_type == "column_terminology":
        # 需求六 业务术语推荐（字段）
        return column_terminology.multiple_match(request)
    elif business_type == "resource_resource":
        # 需求七 相关资产推荐
        return resource_resource.multiple_match(request)
    elif business_type == "resource_terminology":
        # 需求八 业务术语推荐（资产）
        return resource_terminology.multiple_match(request)
    return Response({"code": 404, "msg": "该类型数据推荐正在开发中", "data": ""})


'''
模型和数据初始化总入口
'''


@csrf_exempt
@api_view(http_method_names=['post'])
@permission_classes((permissions.AllowAny,))
def init_model_vector(request):
    parameter = request.data
    business_type = parameter['businessType']
    # 需求1，目录数据推荐初始化
    if business_type == 'catalog_data':
        return catalog_data.init_model_vector(request)
    # 需求2，政务目录数据推荐初始化
    elif business_type == 'item_material':
        return item_material.init_model_vector(request)
    # 需求3，根据数据表字段推荐模型属性
    elif business_type == 'data_model':
        return data_model.init_model_vector(request)
    # 需求4，根据模型属性推荐数据表字段
    elif business_type == 'model_data':
        return model_data.init_model_vector(request)
    elif business_type == "column_meta":
        return column_meta.init_model_vector(request)
    elif business_type == "column_terminology":
        return column_terminology.init_model_vector(request)
    elif business_type == "resource_resource":
        return resource_resource.init_model_vector(request)
    elif business_type == "resource_terminology":
        return resource_terminology.init_model_vector(request)
    return Response({"code": 404, "msg": "该类型数据推荐正在开发中", "data": ""})


'''
数据新增总入口
'''


@csrf_exempt
@api_view(http_method_names=['post'])
@permission_classes((permissions.AllowAny,))
def increment_business_data(request):
    parameter = request.data
    business_type = parameter['businessType']
    # 需求1，目录数据增加
    if business_type == 'catalog_data':
        return catalog_data.increment_data(request)
    # 需求2，政务目录数据增加
    elif business_type == 'item_material':
        return item_material.increment_data(request)
    # 需求3，模型表数据增加
    elif business_type == 'data_model':
        return data_model.increment_data(request)
    # 需求4，数据表数据增加
    elif business_type == 'model_data':
        return model_data.increment_data(request)
    elif business_type == "column_meta":
        return column_meta.increment_data(request)
    elif business_type == "column_terminology":
        return column_terminology.increment_data(request)
    elif business_type == "resource_resource":
        return resource_resource.increment_data(request)
    elif business_type == "resource_terminology":
        return resource_terminology.increment_data(request)
    return Response({"code": 404, "msg": "该类型数据推荐正在开发中", "data": ""})


'''
数据删除总入口
'''


@csrf_exempt
@api_view(http_method_names=['post'])
@permission_classes((permissions.AllowAny,))
def delete_business_data(request):
    parameter = request.data
    business_type = parameter['businessType']
    # 需求1，目录数据删除
    if business_type == 'catalog_data':
        return catalog_data.delete_data(request)
    # 需求2，政务目录数据删除
    elif business_type == 'item_material':
        return item_material.delete_data(request)
    # 需求3，模型表数据删除
    elif business_type == 'data_model':
        return data_model.delete_data(request)
    # 需求4，数据表数据删除
    elif business_type == 'model_data':
        return model_data.delete_data(request)
    elif business_type == "column_meta":
        return column_meta.delete_data(request)
    elif business_type == "column_terminology":
        return column_terminology.delete_data(request)
    elif business_type == "resource_resource":
        return resource_resource.delete_data(request)
    elif business_type == "resource_terminology":
        return resource_terminology.delete_data(request)
    return Response({"code": 404, "msg": "该类型数据推荐正在开发中", "data": ""})
