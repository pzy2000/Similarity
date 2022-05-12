# coding=utf-8
import os

from concurrent import futures
import queue
import gensim
import jieba
import numpy as np
import tensorflow as tf
import torch

import configparser


from demo.settings import DEBUG
from similarity.tools import root_path
from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from similarity.tools import model_dir, data_dir
from similarity.bert_src.similarity_count import BertSim
from .database_get import db

# 默认模型
# model_dir = os.getcwd() + '/similarity/model/'
model_path = model_dir + 'current_model.bin'

# [DEBUG ONLY] 使用limit参数进行快速测试，减少加载时间，减少内存消耗
GENSIM_MODELS_WORD_LIMIT = 20000 if DEBUG and True else None

model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=GENSIM_MODELS_WORD_LIMIT)
dim = len(model.vectors[0])
catalogue_data_path = os.getcwd() + '/similarity/data/政务数据目录编制数据.xlsx'
# 处理完后的目录表路径
exec_catalog_path = os.path.join(data_dir, 'catalog_data.csv')

bert_sim = BertSim()
bert_sim.set_mode(tf.estimator.ModeKeys.PREDICT)
catalogue_data_number = 12000
catalogue_data = []
catalogue_data_vector_department = []
catalogue_data_vector_catalog = []
catalogue_data_vector_catalog_disc = []
catalogue_data_vector_item = []
catalogue_data_vector_item_disc = []
# 目录表信息项tensor的保存路径
catalog_item_tensor_path = os.path.join(data_dir, 'catalog_item.pt')
# 目录表部门tensor的保存路径
catalog_department_tensor_path = os.path.join(data_dir, 'catalog_department.pt')
# 目录表目录项tensor的保存路径
catalog_catalog_tensor_path = os.path.join(data_dir, 'catalog_catalog.pt')
bert_data = {}
query_data = {}
process = 0
percent = [0.4, 0.1, 0, 0.3, 0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
catalogue_data_tensor_department = None
catalogue_data_tensor_catalog = None
catalogue_data_tensor_catalog_disc = None
catalogue_data_tensor_item = None
catalogue_data_tensor_item_disc = None
weight_data = {}
# 数据库读取相关数据
keyword = 'common_data'
read_ini = configparser.ConfigParser()
read_ini.read(os.path.join(root_path, 'config.ini'), encoding='utf-8')

data_col = [int(x) for x in read_ini.get(keyword, 'data_col').split(',')]
table_name = read_ini.get(keyword, 'table_name')
business_type = 'catalog_data'


# database_original_code = []
# database_original_data = []


class RejectQueue(queue.Queue):
    def __init__(self, maxsize=20):
        super(RejectQueue, self).__init__(maxsize=maxsize)

    def put(self, item, block=False, timeout=None):
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        pass
                    else:
                        self._put(item)
                        self.unfinished_tasks += 1
                        self.not_empty.notify()


class ThreadPoolExecutorWithQueueSizeLimit(futures.ThreadPoolExecutor):
    def __init__(self, maxsize=20, *args, **kwargs):
        super(ThreadPoolExecutorWithQueueSizeLimit, self).__init__(*args, **kwargs)
        self._work_queue = RejectQueue(maxsize=maxsize)


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


executor = ThreadPoolExecutorWithQueueSizeLimit(max_workers=5, maxsize=50)
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


def init_model_vector_catalog(request):
    global process
    global catalogue_data_number
    global catalogue_data
    global catalogue_data_vector_department
    global catalogue_data_vector_catalog
    global catalogue_data_vector_catalog_disc
    global catalogue_data_vector_item
    global catalogue_data_vector_item_disc

    global catalogue_data_tensor_department
    global catalogue_data_tensor_catalog
    global catalogue_data_tensor_catalog_disc
    global catalogue_data_tensor_item
    global catalogue_data_tensor_item_disc
    global catalogue_data_path
    parameter = request.data
    # # 目录表路径
    # catalogue_data_path = parameter['filePath']
    # if not os.path.exists(catalogue_data_path):
    #     return Response({"code": 404, "msg": "目录表路径不存在", "data": ""})
    process = 0
    # 重新加载模型
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    process = 0.5
    # 重新缓存向量
    catalogue_data = []
    catalogue_data_vector_department = []
    catalogue_data_vector_catalog = []
    catalogue_data_vector_catalog_disc = []
    catalogue_data_vector_item = []
    catalogue_data_vector_item_disc = []
    # prepare_catalogue_data(path=catalogue_data_path)
    prepare_catalogue_data()
    process = 0.75
    catalogue_data_number = len(catalogue_data)
    for i in range(len(catalogue_data)):
        process = 0.75 + i / (catalogue_data_number * 4)
        data = catalogue_data[i]
        item = data.split(' ')
        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_department.append(s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_catalog.append(s2)

        segment2_1 = jieba.lcut(item[2], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_catalog_disc.append(s2)

        segment2_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_item.append(s2)

        segment2_1 = jieba.lcut(item[4], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_item_disc.append(s2)

    catalogue_data_tensor_department = torch.Tensor(catalogue_data_vector_department).to(device)
    catalogue_data_tensor_catalog = torch.Tensor(catalogue_data_vector_catalog).to(device)
    catalogue_data_tensor_catalog_disc = torch.Tensor(catalogue_data_vector_catalog_disc).to(device)
    catalogue_data_tensor_item = torch.Tensor(catalogue_data_vector_item).to(device)
    catalogue_data_tensor_item_disc = torch.Tensor(catalogue_data_vector_item_disc).to(device)

    bert_data.clear()
    query_data.clear()
    weight_data.clear()
    return Response({"code": 200, "msg": "词模型初始化完成；词向量缓存完成！", "data": ""})


def increment_business_data_catalog(request):
    global catalogue_data_tensor_department
    global catalogue_data_tensor_catalog
    global catalogue_data_tensor_catalog_disc
    global catalogue_data_tensor_item
    global catalogue_data_tensor_item_disc
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        # 加入缓存中
        # tmp = original_data['departmentName'] + ' ' + original_data['catalogName'] + ' ' + \
        #       original_data['infoItemName'] + ' ' + original_data['departmentID'] + ' ' + original_data['catalogID']

        # item = [x.strip() for x in match_str.split('^')]
        # item.append(original_code)
        # item.append(str(original_data).replace(' ', ''))
        # tmp = ' '.join(item)
        if len(match_str.split('^')) != 5:
            return Response({"code": 200, "msg": "新增数据失败，有效数据字段不等于5", "data": ""})
        tmp = ' '.join([x.strip() for x in match_str.split('^')])
        # database_original_code.append(original_code)
        # database_original_data.append(original_data)

        tmp += (' ' + original_code + ' ' + original_data)

        catalogue_data.append(tmp)

        if DEBUG:
            print('增加后：')
            print('catalogue_data：' + str(len(catalogue_data)))
            for i in range(len(catalogue_data)):
                print(catalogue_data[i])

        item = tmp.split(' ')
        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_department.append(s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_catalog.append(s2)

        segment2_1 = jieba.lcut(item[2], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_catalog_disc.append(s2)

        segment2_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_item.append(s2)

        segment2_1 = jieba.lcut(item[4], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        catalogue_data_vector_item_disc.append(s2)

    catalogue_data_tensor_department = torch.Tensor(catalogue_data_vector_department).to(device)
    catalogue_data_tensor_catalog = torch.Tensor(catalogue_data_vector_catalog).to(device)
    catalogue_data_tensor_catalog_disc = torch.Tensor(catalogue_data_vector_catalog_disc).to(device)
    catalogue_data_tensor_item = torch.Tensor(catalogue_data_vector_item).to(device)
    catalogue_data_tensor_item_disc = torch.Tensor(catalogue_data_vector_item_disc).to(device)

    bert_data.clear()
    query_data.clear()
    weight_data.clear()
    return Response({"code": 200, "msg": "新增数据成功！", "data": ""})


def delete_business_data_catalog(request):
    global catalogue_data_tensor_department
    global catalogue_data_tensor_catalog
    global catalogue_data_tensor_catalog_disc
    global catalogue_data_tensor_item
    global catalogue_data_tensor_item_disc
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        # 加入缓存中
        # tmp = original_data['departmentName'] + ' ' + original_data['catalogName'] + ' ' + \
        #       original_data['infoItemName'] + ' ' + original_data['departmentID'] + ' ' + original_data['catalogID']

        tmp = ' '.join([x.strip() for x in match_str.split('^')])
        tmp += (' ' + original_code + ' ' + original_data)

        if DEBUG:
            print('待删除数据：')
            print(tmp)
        # 在目录列表中删除数据
        try:
            catalogue_data.remove(tmp)
        except:
            return Response({"code": 200, "msg": "无该数据！", "data": ""})

        if DEBUG:
            print('删除后：')
            print('catalogue_data：' + str(len(catalogue_data)))
            for i in range(len(catalogue_data)):
                print(catalogue_data[i])

        item = tmp.split(' ')
        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(catalogue_data_vector_department, s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(catalogue_data_vector_catalog, s2)

        segment2_1 = jieba.lcut(item[2], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(catalogue_data_vector_catalog_disc, s2)

        segment2_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(catalogue_data_vector_item, s2)

        segment2_1 = jieba.lcut(item[4], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(catalogue_data_vector_item_disc, s2)

    catalogue_data_tensor_department = torch.Tensor(catalogue_data_vector_department).to(device)
    catalogue_data_tensor_catalog = torch.Tensor(catalogue_data_vector_catalog).to(device)
    catalogue_data_tensor_catalog_disc = torch.Tensor(catalogue_data_vector_catalog_disc).to(device)
    catalogue_data_tensor_item = torch.Tensor(catalogue_data_vector_item).to(device)
    catalogue_data_tensor_item_disc = torch.Tensor(catalogue_data_vector_item_disc).to(device)
    bert_data.clear()
    query_data.clear()
    weight_data.clear()
    return Response({"code": 200, "msg": "删除数据成功！", "data": ""})


def delete_ndarray(with_array_list, array):
    for i in range(len(with_array_list)):
        if all(with_array_list[i] == np.array(array)) == True:
            with_array_list.pop(i)
            break


def catalog_multiple_match(request):
    global percent

    parameter = request.data
    full_data = parameter['data']
    k = parameter['k']
    if k > len(catalogue_data):
        k = len(catalogue_data)
    weight_percent = parameter['percent']
    if len(weight_percent.split(',')) != 5:
        return Response({"code": 404, "msg": "权重配置错误！", "data": ''})
    percent = [float(x) for x in weight_percent.split(',')]

    if len(catalogue_data) == 0:
        return Response({"code": 404, "msg": "数据为空！", "data": ''})
    source_data = []
    for i in range(len(full_data)):
        source_data.append(full_data[i]['matchStr'].replace('^', ' '))
    result = []
    for i in range(len(source_data)):
        res = {}
        data = source_data[i]
        query_id = full_data[i]['id']
        # 字符串匹配
        str_tmp = string_matching(demand_data=data, k=k)
        if len(str_tmp) >= k:
            sim_value = [1] * len(str_tmp)
            result.append(save_result(str_tmp, res, query_id, sim_value))
            continue
        # elif len(str_tmp) > 0 and len(str_tmp) < k:
        #     sim_value = [1] * len(str_tmp)
        #     result.append(save_result(str_tmp, res, query_id, sim_value))

        # 查看查询缓存
        if data in query_data.keys() and weight_data[data] == percent:
            tmp = query_data.get(data)
            if len(tmp) == 2 * k:
                sim_value = tmp[int(len(tmp) / 2):]
                tmp = tmp[0: int(len(tmp) / 2)]
                result.append(save_result(tmp, res, query_id, sim_value))
                continue

        # 查看BERT缓存
        tmp = find_data(demand_data=data, k=k)
        if len(tmp) != 0 and weight_data[data] == percent:
            sim_value = tmp[int(len(tmp) / 2):]
            tmp = tmp[0: int(len(tmp) / 2)]
            result.append(save_result(tmp, res, query_id, sim_value))
            continue

        # 缓存清理FIFO
        if len(bert_data.keys()) >= 10000:
            bert_data.clear()
        if len(query_data.keys()) >= 10000:
            query_data.clear()
        if len(weight_data.keys()) >= 10000:
            weight_data.clear()

        # 词向量匹配
        tmp, sim_value = vector_matching(demand_data=data, k=k)

        if DEBUG:
            print('原来的str_tmp')
            for index in range(len(str_tmp)):
                print(str_tmp[index])

            print('原来的tmp：')
            for index in range(len(tmp)):
                print(tmp[index] + ' : ' + str(sim_value[index]))

        origin_len = len(str_tmp)
        str_tmp += tmp
        str_sim_value = ([1] * origin_len) + sim_value

        if DEBUG:
            print()
            print('增加后的长度：' + str(len(str_tmp)))
            print('增长后的情况：')
            for index in range(len(str_tmp)):
                print(str_tmp[index] + ' : ' + str(str_sim_value[index]))

        for index in range(origin_len):
            while str_tmp[index] in str_tmp[origin_len:]:
                for tmp_index in range(origin_len, len(str_tmp)):
                    if str_tmp[tmp_index] == str_tmp[index]:
                        str_tmp.pop(tmp_index)
                        str_sim_value.pop(tmp_index)
                        break

        # 保证增加后的数据不超过k个
        if len(str_tmp) > k:
            str_tmp = str_tmp[:k]

        if DEBUG:
            print()
            print('删除后的情况：')
            for tmp_index in range(len(str_tmp)):
                print(str_tmp[tmp_index] + ' : ' + str(str_sim_value[tmp_index]))
                # print(str_sim_value[tmp_index])

        # result.append(save_result(tmp, res, query_id, sim_value))
        result.append(save_result(str_tmp, res, query_id, str_sim_value))
        query_data[data] = tmp + sim_value
        weight_data[data] = percent

        # 缓存中不存在, 后台线程缓存
        executor.submit(save_data, data, k)

    return Response({"code": 200, "msg": "查询成功！", "data": result})


def string_matching(demand_data, k):
    res = []

    for data in catalogue_data:
        # if demand_data == tmp_data[0] + ' ' + tmp_data[1] + ' ' + tmp_data[2]:
        tmp_match_str = demand_data.split(' ')
        if tmp_match_str[0] == '' and tmp_match_str[1] == '' and tmp_match_str[3] == '':
            continue
        match_str = tmp_match_str[0] + ' ' + tmp_match_str[1] + ' ' + tmp_match_str[3]

        tmp_database_str = data.split(' ')
        if tmp_database_str[0] == '' and tmp_database_str[1] == '' and tmp_database_str[3] == '':
            continue
        tmp_str = tmp_database_str[0] + ' ' + tmp_database_str[1] + ' ' + tmp_database_str[3]

        if match_str == tmp_str:
            if DEBUG:
                print(111111111)
            res.append(data)
            if len(res) == k:
                break
    return res


def find_data(demand_data, k):
    if demand_data in bert_data.keys():
        tmp = bert_data.get(demand_data)
        if len(tmp) == 2 * k:
            return tmp
    return []


def save_data(demand_data, k):
    return
    sim_words = {}
    item1 = demand_data.split(' ')
    for data in catalogue_data:
        sim = 0
        item2 = data.split(' ')
        sim += bert_sim.predict(item1[0], item2[0])[0][1] * percent[0]
        sim += bert_sim.predict(item1[1], item2[1])[0][1] * percent[1]
        sim += bert_sim.predict(item1[2], item2[2])[0][1] * percent[2]
        sim += bert_sim.predict(item1[3], item2[3])[0][1] * percent[3]
        sim += bert_sim.predict(item1[4], item2[4])[0][1] * percent[4]
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
        elif sim_word[1] < 0:
            sim_word[1] = abs(sim_word[1])
        res.append(sim_word[1])
    bert_data[demand_data] = res


def vector_matching(demand_data, k):
    global catalogue_data_vector_department
    global catalogue_data_vector_catalog
    global catalogue_data_vector_catalog_disc
    global catalogue_data_vector_item
    global catalogue_data_vector_item_disc

    global catalogue_data_tensor_department
    global catalogue_data_tensor_catalog
    global catalogue_data_tensor_catalog_disc
    global catalogue_data_tensor_item
    global catalogue_data_tensor_item_disc
    # 字符串没有匹配项，则会进行向量相似度匹配，筛选前k个
    # sim_words = {}
    item = demand_data.split(' ')

    segment1_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
    s1 = [word_avg(model,  segment1_1)]
    x = torch.Tensor(s1).to(device)
    final_value = tensor_module(catalogue_data_tensor_item, x) * percent[3]

    if item[0] != '':
        segment1_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(catalogue_data_tensor_department, x) * percent[0]

    if item[1] != '':
        segment1_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(catalogue_data_tensor_catalog, x) * percent[1]

    if item[2] != '':
        segment1_1 = jieba.lcut(item[2], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(catalogue_data_tensor_catalog_disc, x) * percent[2]

    if item[4] != '':
        segment1_1 = jieba.lcut(item[4], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(catalogue_data_tensor_item_disc, x) * percent[4]

    # 输出排序并输出top-k的输出
    value, index = torch.topk(final_value, k, dim=0, largest=True, sorted=True)
    sim_index = index.numpy().tolist()
    sim_value = value.numpy().tolist()
    res = []
    res_sim_value = []
    for i in sim_index:
        res.append(catalogue_data[i[0]])
    # print('计算出的匹配值：')
    for i in sim_value:
        # print(i[0])
        if i[0] > 1:
            i[0] = 1.0
        elif i[0] < 0:
            i[0] = abs(i[0])
        res_sim_value.append(i[0])
    return res, res_sim_value


def prepare_catalogue_data():
    global catalogue_data
    global table_name

    re = db.get_data_by_type_v2(data_col, business_type, table_name)

    for i in re:
        catalogue_data.append(' '.join([' '.join([x.strip() for x in i[0].split('^')]), i[1], i[2]]))

    if DEBUG:
        print(catalogue_data)
        print('catalogue_data：' + str(len(catalogue_data)))
        for i in range(len(catalogue_data)):
            print(catalogue_data[i])

    # catalogue_df = pd.DataFrame(catalogue_data)
    # catalogue_df.to_csv(exec_catalog_path, encoding='utf-8_sig', index=False)


def word_avg(word_model, words):  # 对句子中的每个词的词向量简单做平均 作为句子的向量表示
    if len(words) == 0:
        return np.mean([[1e-8] * dim], axis=0)
    vectors = []
    for word in words:
        try:
            vector = word_model.get_vector(word)
            vectors.append(vector)
        except KeyError:
            vectors.append([1e-8] * dim)
            continue
    return np.mean(vectors, axis=0)


def save_result(temp, res, query_id, sim_value):
    single_res = []
    for i in range(len(temp)):
        d = temp[i]
        tmp = d.split(' ')
        # single_res.append({'originalCode': tmp[4], 'originalData': {'departmentName': tmp[0], 'catalogName': tmp[1],
        #                                                             'infoItemName': tmp[2],
        #                                                             'departmentID': tmp[3], 'catalogID': tmp[4]},
        #                    'similarity': sim_value[i]})

        single_res.append({'str': ' '.join(tmp[:5]),
                           'originalCode': tmp[5],
                           'originalData': tmp[6],
                           'similarity': sim_value[i]})

    res['key'] = query_id
    res['result'] = single_res
    return res
