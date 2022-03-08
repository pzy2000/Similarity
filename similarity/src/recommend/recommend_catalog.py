# -*- coding:utf-8 -*-

from similarity.tools import model_dir, data_dir
import jieba
import torch
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from similarity.word2vec_similarity_catalog import save_result, find_data, word_avg
from similarity.word2vec_similarity_catalog import model, device, tensor_module, \
    executor, bert_sim, exec_catalog_path, catalog_item_tensor_path, \
    catalog_department_tensor_path



percent = [0.3, 0.2, 0.2, 0.5]
bert_data = {}
query_data = {}
catalogue_data = []


def catalog_recommend(request):
    global catalogue_data
    parameter = request.data
    full_data = parameter['data']
    k = parameter['k']
    load_catalogue_data()
    if len(catalogue_data) == 0:
        return Response({"code": 404, "msg": "模型和向量未初始化！", "data": ''})
    # 顺序是材料名称-材料描述-材料类型-材料来源部门
    source_data = []
    for i in range(len(full_data)):
        source_data.append(full_data[i]['matchStr'].replace('-', ' '))
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

        # 查看查询缓存
        if data in query_data.keys():
            tmp = query_data.get(data)
            if len(tmp) == 2 * k:
                sim_value = tmp[int(len(tmp) / 2):]
                tmp = tmp[0: int(len(tmp) / 2)]
                result.append(save_result(tmp, res, query_id, sim_value))
                continue

        # 查看BERT缓存
        tmp = find_data(demand_data=data, k=k)
        if len(tmp) != 0:
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


def load_catalogue_data():
    global catalogue_data
    catalogue_df = pd.read_csv(exec_catalog_path, encoding='utf-8')
    catalogue_data = [str(x[0]) for x in catalogue_df.values]
    # print(catalogue_data)

def string_matching(demand_data, k):
    res = []
    for data in catalogue_data:
        tmp_data = data.split(' ')
        if demand_data == tmp_data[0] + ' ' + tmp_data[1] + ' ' + tmp_data[2]:
            res.append(data)
            if len(res) == k:
                break
    return res

def vector_matching(demand_data, k):
    # 字符串没有匹配项，则会进行向量相似度匹配，筛选前k个
    # sim_words = {}
    item = demand_data.split(' ')

    # 事项
    segment1_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
    s1 = [word_avg(model, segment1_1)]
    x = torch.Tensor(s1).to(device)
    final_value = tensor_module(torch.load(catalog_item_tensor_path), x) * percent[0]

    # 部门
    if item[3] != '':
        segment1_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(torch.load(catalog_department_tensor_path), x) * percent[3]

    # # 材料描述、材料类型
    # if item[1] != '':
    #     segment1_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
    #     s1 = [word_avg(model, segment1_1)]
    #     x = torch.Tensor(s1).to(device)
    #     final_value += tensor_module(catalogue_data_tensor_catalog, x) * percent[1]

    # 输出排序并输出top-k的输出
    value, index = torch.topk(final_value, k, dim=0, largest=True, sorted=True)
    sim_index = index.numpy().tolist()
    sim_value = value.numpy().tolist()
    res = []
    res_sim_value = []
    for i in sim_index:
        res.append(catalogue_data[i[0]])
    for i in sim_value:
        if i[0] > 1:
            i[0] = 1.0
        res_sim_value.append(i[0])
    return res, res_sim_value

def save_data(demand_data, k):
    sim_words = {}
    # 部门-目录-信息项
    # 输入字符：材料名称-材料描述-材料类型-材料来源部门
    item1 = demand_data.split(' ')
    for data in catalogue_data:
        sim = 0
        item2 = data.split(' ')
        # sim += bert_sim.predict(item1[0], item2[0])[0][1] * percent[0]
        # sim += bert_sim.predict(item1[1], item2[1])[0][1] * percent[1]
        # sim += bert_sim.predict(item1[2], item2[2])[0][1] * percent[2]
        # 信息项
        sim += bert_sim.predict(item1[0], item2[2])[0][1] * percent[0]
        # 部门
        sim += bert_sim.predict(item1[3], item2[0])[0][1] * percent[3]

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
    sim_words = sorted(sim_words.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for sim_word in sim_words:
        res.append(sim_word[0])
    for sim_word in sim_words:
        if sim_word[1] > 1:
            sim_word[1] = 1.0
        res.append(sim_word[1])
    bert_data[demand_data] = res
