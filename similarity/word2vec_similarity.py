# coding=utf-8
import os
from concurrent.futures import ThreadPoolExecutor

import gensim
import jieba
import numpy as np
import tensorflow as tf
import torch
import xlrd
from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from similarity.bert_src.similarity_count import BertSim

# 默认模型
model_dir = os.getcwd() + '/similarity/model/'
model_path = model_dir + 'current_model.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
dim = len(model.vectors[0])
catalogue_data_path = os.getcwd() + '/similarity/data/政务数据目录编制数据.xlsx'
bert_sim = BertSim()
bert_sim.set_mode(tf.estimator.ModeKeys.PREDICT)
catalogue_data_number = 12000
catalogue_data = []
catalogue_data_vector = []
bert_data = {}
query_data = {}
process = 0
executor = ThreadPoolExecutor(max_workers=5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
catalogue_data_tensor = None


class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, x1, x2):
        x2 = x2.t()
        x = x1.mm(x2)

        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)

        final = x.mul(1 / x_frobenins)
        return final


tensor_module = CosineSimilarity().to(device)


@csrf_exempt
@api_view(http_method_names=['get'])  # 只允许get
@permission_classes((permissions.AllowAny,))
def get_state(request):
    if process == 0:
        return Response({"code": 200, "msg": "模型和向量未初始化！", "data": ""})
    if process > 0.99:
        return Response({"code": 200, "msg": "模型和向量初始化完成！", "data": ""})
    return Response({"code": 200, "msg": "模型和向量初始化中！", "data": process})


@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def init_model_vector(request):
    global process
    global catalogue_data_number
    global catalogue_data
    global catalogue_data_vector
    global catalogue_data_tensor
    global catalogue_data_path
    parameter = request.data
    # 目录表路径
    catalogue_data_path = parameter['catalogue_path']
    if not os.path.exists(catalogue_data_path):
        return Response({"code": 404, "msg": "目录表路径不存在", "data": ""})
    process = 0
    # 重新加载模型
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    process = 0.5
    # 重新缓存向量
    catalogue_data = []
    catalogue_data_vector = []
    prepare_catalogue_data(path=catalogue_data_path)
    process = 0.75
    catalogue_data_number = len(catalogue_data)
    for i in range(len(catalogue_data)):
        process = 0.75 + i / (catalogue_data_number * 4)
        data = catalogue_data[i]
        item = data.split(' ')
        segment2_1 = jieba.lcut(item[1] + item[2], cut_all=True, HMM=True)
        segment2_2 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1, segment2_2)
        catalogue_data_vector.append(s2)
    catalogue_data_tensor = torch.Tensor(catalogue_data_vector).to(device)
    return Response({"code": 200, "msg": "词模型初始化完成；词向量缓存完成！", "data": ""})


@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def multiple_match(request):
    parameter = request.data
    full_data = parameter['data']
    k = parameter['k']
    if len(catalogue_data) == 0:
        return Response({"code": 404, "msg": "模型和向量未初始化！", "data": ''})
    source_data = []
    for i in range(len(full_data)):
        source_data.append(full_data[i]['departmentName'] + ' ' + full_data[i]['catalogName'] + ' '
                           + full_data[i]['infoItemName'])
    result = []
    for i in range(len(source_data)):
        res = {}
        data = source_data[i]
        query_id = full_data[i]['id']
        # 字符串匹配
        tmp = string_matching(demand_data=data, k=k)
        if len(tmp) != 0:
            sim_value = [1] * len(tmp)
            result.append(save_result(tmp, res, query_id, sim_value))
            continue

        # 查看BERT缓存
        tmp = find_data(demand_data=data, k=k)
        if len(tmp) != 0:
            sim_value = tmp[int(len(tmp) / 2):]
            tmp = tmp[0: int(len(tmp) / 2)]
            result.append(save_result(tmp, res, query_id, sim_value))
            continue

        # 查看查询缓存
        if data in query_data.keys():
            tmp = query_data.get(data)
            if len(tmp) == k:
                sim_value = tmp[int(len(tmp) / 2):]
                tmp = tmp[0: int(len(tmp) / 2)]
                result.append(save_result(tmp, res, query_id, sim_value))
                continue

        # 缓存清理FIFO
        if len(bert_data.keys()) >= 10000:
            bert_data.clear()
        if len(query_data.keys()) >= 10000:
            query_data.clear()

        # 词向量匹配
        tmp, sim_value = vector_matching(demand_data=data, k=k)
        result.append(save_result(tmp, res, query_id, sim_value))
        query_data[data] = tmp + sim_value

        # 缓存中不存在, 后台线程缓存
        executor.submit(save_data, data, k)

    return Response({"code": 200, "msg": "查询成功！", "data": result})


def string_matching(demand_data, k):
    res = []
    for data in catalogue_data:
        tmp_data = data.split(' ')
        if demand_data == tmp_data[0] + ' ' + tmp_data[1] + ' ' + tmp_data[2]:
            res.append(data)
            if len(res) == k:
                break
    return res


def find_data(demand_data, k):
    if demand_data in bert_data.keys():
        tmp = bert_data.get(demand_data)
        if len(tmp) == k:
            return tmp
    return []


def save_data(demand_data, k):
    sim_words = {}
    for data in catalogue_data:
        sim = bert_sim.predict(data, demand_data)[0][1]
        if len(sim_words) < k:
            sim_words[data] = sim
        else:
            min_sim = min(sim_words.values())
            if sim > min_sim:
                for key in list(sim_words.keys()):
                    if sim_words.get(key) == min_sim:
                        # 替换
                        del sim_words[key]
                        sim_words[data] = sim
                        break
    res = []
    for sim_word in sim_words:
        res.append(sim_word)
    for sim_word in sim_words:
        res.append(sim_words.get(sim_word))
    bert_data[demand_data] = res


def vector_matching(demand_data, k):
    # 字符串没有匹配项，则会进行向量相似度匹配，筛选前k个
    # sim_words = {}
    item = demand_data.split(' ')
    segment1_1 = jieba.lcut(item[1] + item[2], cut_all=True, HMM=True)
    segment1_2 = jieba.lcut(item[0], cut_all=True, HMM=True)
    s1 = [word_avg(model, segment1_1, segment1_2)]
    x = torch.Tensor(s1).to(device)
    final_value = tensor_module(catalogue_data_tensor, x)

    # 输出排序并输出top-k的输出
    value, index = torch.topk(final_value, k, dim=0, largest=True, sorted=True)
    sim_index = index.numpy().tolist()
    sim_value = value.numpy().tolist()
    res = []
    res_sim_value = []
    for i in sim_index:
        res.append(catalogue_data[i[0]])
    for i in sim_value:
        res_sim_value.append(i[0])
    return res, res_sim_value


def prepare_catalogue_data(path):
    # 打开excel
    wb = xlrd.open_workbook(path)
    # 按工作簿定位工作表
    sh = wb.sheet_by_name('信息资源导入模板')
    row_number = sh.nrows
    for i in range(2, row_number):
        catalogue_data.append(sh.cell(i, 0).value + ' ' + sh.cell(i, 3).value + ' ' + sh.cell(i, 11).value + ' ' +
                              sh.cell(i, 6).value + ' ' + sh.cell(i, 15).value)


def word_avg(word_model, words, last_words):  # 对句子中的每个词的词向量简单做平均 作为句子的向量表示
    vectors = []
    for word in words:
        try:
            vector = word_model.get_vector(word)
            vectors.append(vector)
        except KeyError:
            vectors.append([1e-8] * dim)
            continue
    for word in last_words:
        try:
            vector = word_model.get_vector(word)
            vectors.append(vector * 3)
        except KeyError:
            vectors.append([1e-8] * dim)
            continue
    return np.mean(vectors, axis=0)


def save_result(temp, res, query_id, sim_value):
    single_res = []
    for i in range(len(temp)):
        d = temp[i]
        tmp = d.split(' ')
        single_res.append({'departmentName': tmp[0], 'catalogName': tmp[1], 'infoItemName': tmp[2],
                           'departmentID': tmp[3], 'catalogID': tmp[4], 'sim': sim_value[i]})
    res['key'] = query_id
    res['result'] = single_res
    return res

