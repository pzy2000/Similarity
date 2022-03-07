# coding=utf-8

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import queue
import gensim
import jieba
import numpy as np
import tensorflow as tf
import torch
import xlrd
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from similarity.tools import model_dir, data_dir
from similarity.bert_src.similarity_count import BertSim
from similarity.word2vec_similarity_catalog import word_avg, device, model

model_data_path = os.path.join(data_dir + '/01人口模型汇总_v1.0.xlsx')
# 处理完后的目录表路径
exec_model_path = os.path.join(data_dir, '/model_data.csv')
model_data = []
model_data_number = 4900
model_data__vector_modelName = []
model_data__vector_valueName = []
model_data_tensor_modelName = None
model_data_tensor_valueName = None
# 目录表信息项tensor的保存路径
model_modelName_tensor_path = os.path.join(data_dir, 'model_modelName.pt')
# 目录表部门tensor的保存路径
model_valueName_tensor_path = os.path.join(data_dir, 'model_valueName.pt')
bert_data = {}
query_data = {}



def init_model_vector_model(request):
    global process

    global model_data_path
    global model_data
    global model_data_number
    global model_data__vector_modelName
    global model_data__vector_valueName
    global model_data_tensor_modelName
    global model_data_tensor_valueName

    parameter = request.data
    # 目录表路径
    model_data_path = parameter['filePath']
    if not os.path.exists(model_data_path):
        return Response({"code": 404, "msg": "模型表路径不存在", "data": ""})
    process = 0
    # 重新加载模型
    model = gensim.models.KeyedVectors.load_word2vec_format(model_data_path, binary=True)
    process = 0.5
    # 重新缓存向量
    model_data = []
    model_data_vector_modelName = []
    model_data_vector_valueName = []
    prepare_model_data(path=model_data_path)
    process = 0.75
    model_data_number = len(model_data)
    for i in range(len(model_data)):
        process = 0.75 + i / (model_data_number * 4)
        data = model_data[i]
        item = data.split(' ')
        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data_vector_modelName.append(s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data_vector_valueName.append(s2)
    model_data_tensor_modelName = torch.Tensor(model_data_vector_modelName).to(device)
    model_data_tensor_valueName = torch.Tensor(model_data__vector_valueName).to(device)

    torch.save(model_data_tensor_modelName, model_modelName_tensor_path)
    torch.save(model_data_tensor_valueName, model_valueName_tensor_path)

    bert_data.clear()
    query_data.clear()
    return Response({"code": 200, "msg": "词模型初始化完成；词向量缓存完成！", "data": ""})

def prepare_model_data(path):
    # 打开excel
    wb = xlrd.open_workbook(path)
    # 按工作簿定位工作表
    sh = wb.sheet_by_name('业务层模型结构')
    row_number = sh.nrows
    for i in range(1, row_number):
        model_data.append(sh.cell(i, 2).value + ' ' + sh.cell(i, 5).value)
    # print(model_data)
    model_df = pd.DataFrame(model_data)
    model_df.to_csv(exec_model_path, encoding='utf-8_sig', index=False)

def increment_business_data_model(request):
    global model_data_tensor_modelName
    global model_data_tensor_valueName
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        match_str.replace('-', ' ')
        # 加入缓存中
        # tmp = original_data['departmentName'] + ' ' + original_data['catalogName'] + ' ' + \
        #       original_data['infoItemName'] + ' ' + original_data['departmentID'] + ' ' + original_data['catalogID']
        model_data.append(match_str)
        item = match_str.split(' ')
        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data__vector_modelName.append(s2)
        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data__vector_valueName.append(s2)
    model_data_tensor_modelName = torch.Tensor(model_data__vector_modelName).to(device)
    model_data_tensor_valueName = torch.Tensor(model_data__vector_valueName).to(device)
    bert_data.clear()
    query_data.clear()
    return Response({"code": 200, "msg": "新增数据成功！", "data": ""})

def delete_business_data_model(request):
    global model_data_tensor_modelName
    global model_data_tensor_valueName
    global model_data__vector_modelName
    global model_data__vector_valueName
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        # 加入缓存中
        # tmp = original_data['departmentName'] + ' ' + original_data['catalogName'] + ' ' + \
        #       original_data['infoItemName'] + ' ' + original_data['departmentID'] + ' ' + original_data['catalogID']
        match_str.replace('-', ' ')
        # 在目录列表中删除数据
        try:
            model_data.remove(match_str)
        except:
            return Response({"code": 200, "msg": "无该数据！", "data": ""})
        item = match_str.split(' ')
        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(model_data__vector_modelName, s2)
        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(model_data__vector_valueName, s2)
    model_data_tensor_modelName = torch.Tensor(model_data__vector_modelName).to(device)
    model_data_tensor_valueName = torch.Tensor(model_data__vector_valueName).to(device)
    bert_data.clear()
    query_data.clear()
    return Response({"code": 200, "msg": "删除数据成功！", "data": ""})

def delete_ndarray(with_array_list, array):
    for i in range(len(with_array_list)):
        if all(with_array_list[i] == np.array(array)) == True:
            with_array_list.pop(i)
            break