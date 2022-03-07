# coding=utf-8

from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from .word2vec_similarity_catalog import catalog_multiple_match, init_model_vector_catalog, \
    increment_business_data_catalog, delete_business_data_catalog
from rest_framework.response import Response
from similarity.src.recommend.recommend_catalog import catalog_recommend
from similarity.src.process.data_to_model import init_model_vector_model, \
    increment_business_data_model, delete_business_data_model

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
        return init_model_vector_catalog(request)
    # 需求3，根据数据表字段推荐模型属性
    elif business_type == 'data_model':
        return init_model_vector_model(request)
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
        return increment_business_data_catalog(request)
    # 需求3，模型表数据增加
    elif business_type == 'item_material':
        return increment_business_data_model(request)
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
        return delete_business_data_catalog(request)
    # 需求3，模型表数据删除
    elif business_type == 'item_material':
        return delete_business_data_model(request)
    return Response({"code": 404, "msg": "该类型数据推荐正在开发中", "data": ""})
