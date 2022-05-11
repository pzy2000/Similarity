# coding=utf-8

from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from .word2vec_similarity_catalog import catalog_multiple_match, init_model_vector_catalog, \
    increment_business_data_catalog, delete_business_data_catalog
from rest_framework.response import Response
from similarity.src.recommend.recommend_catalog import catalog_recommend, \
    init_model_vector_material, increment_business_data_material, delete_business_data_material
from similarity.src.process.data_to_model import init_model_vector_model, \
    increment_business_data_model, delete_business_data_model, data2model_recommend
from similarity.src.process.model_to_data import init_model_vector_data, \
    increment_business_model_data, delete_business_model_data, model2data_recommend
# =============similarity2===============
from similarity2.src import column_meta, column_terminology, resource_resource

'''
数据推荐总入口，实现内容包括：
1. 需求一
2. 需求二
3. 需求三
4. 需求四
'''


@csrf_exempt
@api_view(http_method_names=['post'])
@permission_classes((permissions.AllowAny,))
def multiple_match(request):
    parameter = request.data
    business_type = parameter['businessType']
    if business_type == 'catalog_data':
        # 需求一，目录数据推荐
        return catalog_multiple_match(request)
    elif business_type == 'item_material':
        # 需求二，给定事项材料关联目录
        return catalog_recommend(request)
    elif business_type == 'data_model':
        # 需求三，根据数据表字段推荐模型属性
        return data2model_recommend(request)
    elif business_type == 'model_data':
        # 需求四，根据模型表属性推荐数据字段
        return model2data_recommend(request)
    elif business_type == "column_meta":
        # 需求五，推荐数据元
        return column_meta.multiple_match(request)
    elif business_type == "column_terminology":
        # 需求五，推荐数据元
        return column_terminology.multiple_match(request)
    elif business_type == "resource_resource":
        return resource_resource.multiple_match(request)
    else:
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
        return init_model_vector_catalog(request)
    # 需求2，政务目录数据推荐初始化
    elif business_type == 'item_material':
        return init_model_vector_material(request)
    # 需求3，根据数据表字段推荐模型属性
    elif business_type == 'data_model':
        return init_model_vector_model(request)
    # 需求4，根据模型属性推荐数据表字段
    elif business_type == 'model_data':
        return init_model_vector_data(request)
    elif business_type == "column_meta":
        return column_meta.init_model_vector(request)
    elif business_type == "column_terminology":
        return column_terminology.init_model_vector(request)
    elif business_type == "resource_resource":
        return resource_resource.init_model_vector(request)
    else:
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
        return increment_business_data_catalog(request)
    # 需求2，政务目录数据增加
    elif business_type == 'item_material':
        return increment_business_data_material(request)
    # 需求3，模型表数据增加
    elif business_type == 'data_model':
        return increment_business_data_model(request)
    # 需求4，数据表数据增加
    elif business_type == 'model_data':
        return increment_business_model_data(request)
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
        return delete_business_data_catalog(request)
    # 需求2，政务目录数据删除
    elif business_type == 'item_material':
        return delete_business_data_material(request)
    # 需求3，模型表数据删除
    elif business_type == 'data_model':
        return delete_business_data_model(request)
    # 需求4，数据表数据删除
    elif business_type == 'model_data':
        return delete_business_model_data(request)
    return Response({"code": 404, "msg": "该类型数据推荐正在开发中", "data": ""})
