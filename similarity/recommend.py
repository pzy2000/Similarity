# coding=utf-8

from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from .word2vec_similarity import multiple_match
from rest_framework.response import Response

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
def recommend(request):
    parameter = request.data
    recommend_type = parameter['type']
    if recommend_type == 1:
        # 需求一
        multiple_match(request)
    else:
        Response({"code": 404, "msg": "该类型数据推荐正在开发中", "data": ""})
