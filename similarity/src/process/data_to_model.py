# coding=utf-8

import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
import gensim
import jieba
import torch
import xlrd
import configparser
import pandas as pd
from similarity.tools import root_path
from rest_framework.response import Response
from similarity.tools import model_dir, data_dir
from similarity.word2vec_similarity_catalog import find_data, word_avg
from similarity.word2vec_similarity_catalog import device, model, delete_ndarray, \
    bert_sim, executor, tensor_module, model_path
from similarity.database_get import db

data_model_path = os.path.join(data_dir, '01人口模型汇总_v1.0.xlsx')
# 处理完后的目录表路径
exec_model_path = os.path.join(data_dir, 'data_model.csv')
data_model = []
data_model_number = 4900
data_model_vector_department = []
data_model_vector_modelName = []
data_model_vector_modelName_disc = []
data_model_vector_valueName = []
data_model_vector_valueName_disc = []

data_model_tensor_department = None
data_model_tensor_modelName = None
data_model_tensor_modelName_disc = None
data_model_tensor_valueName = None
data_model_tensor_valueName_disc = None
# 模型表模型名称tensor的保存路径
model_modelName_tensor_path = os.path.join(data_dir, 'model_modelName.pt')
# 模型表属性名称tensor的保存路径
model_valueName_tensor_path = os.path.join(data_dir, 'model_valueName.pt')
bert_data = {}
query_data = {}
weight_data = {}
percent = [0.4, 0.1, 0, 0.3, 0]

# keyword = 'data_model'
# read_ini = configparser.ConfigParser()
# read_ini.read(os.path.join(root_path,'config.ini'), encoding='utf-8')
#
# data_col = [int(x) for x in read_ini.get(keyword, 'data_col').split(',')]
# table_name = read_ini.get(keyword, 'table_name')

# 数据库读取相关数据
keyword = 'common_data'
read_ini = configparser.ConfigParser()
read_ini.read(os.path.join(root_path, 'config.ini'), encoding='utf-8')

data_col = [int(x) for x in read_ini.get(keyword, 'data_col').split(',')]
table_name = read_ini.get(keyword, 'table_name')
business_type = 'data_model'


def init_model_vector_model(request):
    global process
    global data_model_path
    global data_model
    global data_model_number
    global data_model_vector_department
    global data_model_vector_modelName
    global data_model_vector_modelName_disc
    global data_model_vector_valueName
    global data_model_vector_valueName_disc

    global data_model_tensor_department
    global data_model_tensor_modelName
    global data_model_tensor_modelName_disc
    global data_model_tensor_valueName
    global data_model_tensor_valueName_disc

    parameter = request.data
    # # 目录表路径
    # data_model_path = parameter['filePath']
    # if not os.path.exists(data_model_path):
    #     return Response({"code": 404, "msg": "模型表路径不存在", "data": ""})
    process = 0
    # 重新加载模型
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    process = 0.5
    # 重新缓存向量
    data_model = []
    data_model_vector_department = []
    data_model_vector_modelName = []
    data_model_vector_modelName_disc = []
    data_model_vector_valueName = []
    data_model_vector_valueName_disc = []
    # prepare_data_model(path=data_model_path)
    prepare_data_model()
    process = 0.75
    data_model_number = len(data_model)
    for i in range(len(data_model)):
        process = 0.75 + i / (data_model_number * 4)
        data = data_model[i]
        item = data.split(' ')

        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_department.append(s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_modelName.append(s2)

        segment2_1 = jieba.lcut(item[2], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_modelName_disc.append(s2)

        segment2_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_valueName.append(s2)

        segment2_1 = jieba.lcut(item[4], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_valueName_disc.append(s2)
    data_model_tensor_department = torch.Tensor(data_model_vector_department).to(device)
    data_model_tensor_modelName = torch.Tensor(data_model_vector_modelName).to(device)
    data_model_tensor_modelName_disc = torch.Tensor(data_model_vector_modelName_disc).to(device)
    data_model_tensor_valueName = torch.Tensor(data_model_vector_valueName).to(device)
    data_model_tensor_valueName_disc = torch.Tensor(data_model_vector_valueName_disc).to(device)

    bert_data.clear()
    query_data.clear()
    weight_data.clear()
    return Response({"code": 200, "msg": "词模型初始化完成；词向量缓存完成！", "data": ""})


def prepare_data_model():
    global data_model
    global table_name
    # # 打开excel
    # wb = xlrd.open_workbook(path)
    # # 按工作簿定位工作表
    # sh = wb.sheet_by_name('业务层模型结构')
    # row_number = sh.nrows
    # for i in range(1, row_number):
    #     data_model.append(sh.cell(i, 2).value + ' ' + sh.cell(i, 5).value)
    # # print(model_data)

    # re = db.get_colum_by_num(data_col, table_name)
    # for i in re:
    #     while (None in i):
    #         i[i.index(None)] = '*'
    #     data_model.append(' '.join(i))
    #
    # model_df = pd.DataFrame(data_model)
    # model_df.to_csv(exec_model_path, encoding='utf-8_sig', index=False)

    re = db.get_data_by_type_v2(data_col, business_type, table_name)
    for i in re:
        data_model.append(' '.join([i[0].replace('-', ' '), i[1], i[2]]))

    # print('data_model：' + str(len(data_model)))
    # for i in range(len(data_model)):
    #     print(data_model[i])

    # model_df = pd.DataFrame(data_model)
    # model_df.to_csv(exec_model_path, encoding='utf-8_sig', index=False)


def increment_business_data_model(request):
    global data_model_vector_department
    global data_model_vector_modelName
    global data_model_vector_modelName_disc
    global data_model_vector_valueName
    global data_model_vector_valueName_disc

    global data_model_tensor_department
    global data_model_tensor_modelName
    global data_model_tensor_modelName_disc
    global data_model_tensor_valueName
    global data_model_tensor_valueName_disc
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        # match_str = match_str.replace('-', ' ')
        # 加入缓存中
        # tmp = original_data['departmentName'] + ' ' + original_data['catalogName'] + ' ' + \
        #       original_data['infoItemName'] + ' ' + original_data['departmentID'] + ' ' + original_data['catalogID']

        # item = match_str.split('-')
        # data_model.append(item[1] + ' ' + item[3])

        if len(match_str.split('-')) != 5:
            return Response({"code": 200, "msg": "新增数据失败，有效数据字段不等于5", "data": ""})

        tmp = ' '.join(match_str.split('-'))

        tmp += (' ' + original_code + ' ' + original_data)
        data_model.append(tmp)

        # print('增加后：')
        # print('data_model：' + str(len(data_model)))
        # for i in range(len(data_model)):
        #     print(data_model[i])

        item = tmp.split(' ')
        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_department.append(s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_modelName.append(s2)

        segment2_1 = jieba.lcut(item[2], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_modelName_disc.append(s2)

        segment2_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_valueName.append(s2)

        segment2_1 = jieba.lcut(item[4], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        data_model_vector_valueName_disc.append(s2)

    data_model_tensor_department = torch.Tensor(data_model_vector_department).to(device)
    data_model_tensor_modelName = torch.Tensor(data_model_vector_modelName).to(device)
    data_model_tensor_modelName_disc = torch.Tensor(data_model_vector_modelName_disc).to(device)
    data_model_tensor_valueName = torch.Tensor(data_model_vector_valueName).to(device)
    data_model_tensor_valueName_disc = torch.Tensor(data_model_vector_valueName_disc).to(device)
    bert_data.clear()
    query_data.clear()
    weight_data.clear()
    return Response({"code": 200, "msg": "新增数据成功！", "data": ""})


def delete_business_data_model(request):
    global data_model_tensor_department
    global data_model_tensor_modelName
    global data_model_tensor_modelName_disc
    global data_model_tensor_valueName
    global data_model_tensor_valueName_disc

    global data_model_vector_department
    global data_model_vector_modelName
    global data_model_vector_modelName_disc
    global data_model_vector_valueName
    global data_model_vector_valueName_disc
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        # 加入缓存中
        # tmp = original_data['departmentName'] + ' ' + original_data['catalogName'] + ' ' + \
        #       original_data['infoItemName'] + ' ' + original_data['departmentID'] + ' ' + original_data['catalogID']
        # match_str = match_str.replace('-', ' ')

        tmp = ' '.join(match_str.split('-'))
        tmp += (' ' + original_code + ' ' + original_data)
        # print('待删除数据：')
        # print(tmp)

        # 在目录列表中删除数据
        try:
            data_model.remove(tmp)
        except:
            return Response({"code": 200, "msg": "无该数据！", "data": ""})

        # print('删除后：')
        # print('data_model：' + str(len(data_model)))
        # for i in range(len(data_model)):
        #     print(data_model[i])

        item = match_str.split('-')
        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(data_model_vector_department, s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(data_model_vector_modelName, s2)

        segment2_1 = jieba.lcut(item[2], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(data_model_vector_modelName_disc, s2)

        segment2_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(data_model_vector_valueName, s2)

        segment2_1 = jieba.lcut(item[4], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(data_model_vector_valueName_disc, s2)

    data_model_tensor_department = torch.Tensor(data_model_vector_department).to(device)
    data_model_tensor_modelName = torch.Tensor(data_model_vector_modelName).to(device)
    data_model_tensor_modelName_disc = torch.Tensor(data_model_vector_modelName_disc).to(device)
    data_model_tensor_valueName = torch.Tensor(data_model_vector_valueName).to(device)
    data_model_tensor_valueName_disc = torch.Tensor(data_model_vector_valueName_disc).to(device)

    bert_data.clear()
    query_data.clear()
    weight_data.clear()
    return Response({"code": 200, "msg": "删除数据成功！", "data": ""})


def data2model_recommend(request):
    global data_model
    global percent
    global weight_data
    parameter = request.data
    full_data = parameter['data']
    k = parameter['k']
    if k > len(data_model):
        k = len(data_model)
    weight_percent = parameter['percent']
    if len(weight_percent.split(',')) != 5:
        return Response({"code": 404, "msg": "权重配置错误！", "data": ''})
    percent = [float(x) for x in weight_percent.split(',')]
    # load_model_data()
    if len(data_model) == 0:
        return Response({"code": 404, "msg": "数据为空！", "data": ''})
    # 顺序是模型表名-字段名
    source_data = []
    for i in range(len(full_data)):
        source_data.append(full_data[i]['matchStr'].replace('-', ' '))
    result = []
    tick = 0
    for i in range(len(source_data)):
        res = {}
        data = source_data[i]
        query_id = full_data[i]['id']
        # 字符串匹配
        str_tmp = string_matching(demand_data=data, k=k)
        if len(str_tmp) >= k:
            sim_value = [1] * len(str_tmp)
            # print("用了字符串匹配")
            result.append(save_result(str_tmp, res, query_id, sim_value))
            continue

        # 查看查询缓存
        if data in query_data.keys():
            tmp = query_data.get(data)
            if len(tmp) == 2 * k and weight_data[data] == percent:
                sim_value = tmp[int(len(tmp) / 2):]
                tmp = tmp[0: int(len(tmp) / 2)]
                # print("用了查询缓存")
                result.append(save_result(tmp, res, query_id, sim_value))
                continue

        # 查看BERT缓存
        tmp = find_data(demand_data=data, k=k)
        if len(tmp) != 0 and weight_data[data] == percent:
            sim_value = tmp[int(len(tmp) / 2):]
            tmp = tmp[0: int(len(tmp) / 2)]
            print("用了BERT缓存")
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
        # print("用了词向量匹配")
        # print('原来的str_tmp')
        # for index in range(len(str_tmp)):
        #     print(str_tmp[index])
        #
        # print('原来的tmp：')
        # for index in range(len(tmp)):
        #     print(tmp[index] + ' : ' + str(sim_value[index]))

        oringi_len = len(str_tmp)
        str_tmp += tmp
        str_sim_value = ([1] * oringi_len) + sim_value

        # print()
        # print('增加后的长度：' + str(len(str_tmp)))
        # print('增长后的情况：')
        # for index in range(len(str_tmp)):
        #     print(str_tmp[index] + ' : ' + str(str_sim_value[index]))

        # for index in range(oringi_len):
        #     if str_tmp[index] == str_tmp[0]:
        #         str_tmp.pop(oringi_len)
        #         str_sim_value.pop(oringi_len)

        for index in range(oringi_len):
            while str_tmp[index] in str_tmp[oringi_len:]:
                for tmp_index in range(oringi_len, len(str_tmp)):
                    if str_tmp[tmp_index] == str_tmp[index]:
                        str_tmp.pop(tmp_index)
                        str_sim_value.pop(tmp_index)
                        break
            # if str_tmp[index] == str_tmp[0]:
            #     str_tmp.pop(oringi_len)
            #     str_sim_value.pop(oringi_len)

        # 保证增加后的数据不超过k个
        if len(str_tmp) > k:
            str_tmp = str_tmp[:k]

        # print()
        # print('删除后的情况：')
        # for tmp_index in range(len(str_tmp)):
        #     print(str_tmp[tmp_index] + ' : ' + str(str_sim_value[tmp_index]))

        # result.append(save_result(tmp, res, query_id, sim_value))
        result.append(save_result(str_tmp, res, query_id, str_sim_value))
        query_data[data] = tmp + sim_value
        weight_data[data] = percent

        # 缓存中不存在, 后台线程缓存
        executor.submit(save_data, data, k)
        tick = 1
        return data2model_recommend(request)

    return Response({"code": 200, "msg": "查询成功！", "data": result})


def string_matching(demand_data, k):
    # res = []
    # for data in data_model:
    #     tmp_data = data.split(' ')
    #     tmp_demand_data = demand_data.split(' ')
    #     if tmp_demand_data[1] + ' ' + tmp_demand_data[3] == tmp_data[0] + ' ' + tmp_data[1]:
    #         res.append(data)
    #         if len(res) == k:
    #             break
    # return res

    res = []
    # print('data_len：' + str(len(data_model)))

    # for i in range(len(data_model)):
    #     print(data_model[i])

    for data in data_model:
        tmp_match_str = demand_data.split(' ')
        match_str = tmp_match_str[0] + ' ' + tmp_match_str[1] + ' ' + tmp_match_str[3]

        tmp_database_str = data.split(' ')
        tmp_str = tmp_database_str[0] + ' ' + tmp_database_str[1] + ' ' + tmp_database_str[3]

        if match_str == tmp_str:
            # print(111111111)
            res.append(data)
            if len(res) == k:
                break
    return res


def vector_matching(demand_data, k):
    # 字符串没有匹配项，则会进行向量相似度匹配，筛选前k个
    # sim_words = {}
    item = demand_data.split(' ')
    # 输入的数据表字段名
    segment1_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
    s1 = [word_avg(model, segment1_1)]
    x = torch.Tensor(s1).to(device)
    final_value = tensor_module(data_model_tensor_valueName, x) * percent[3]

    if item[0] != '':
        segment1_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(data_model_tensor_department, x) * percent[0]

    if item[1] != '':
        segment1_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(data_model_tensor_modelName, x) * percent[1]

    if item[2] != '':
        segment1_1 = jieba.lcut(item[2], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(data_model_tensor_modelName_disc, x) * percent[2]

    '''if item[3] != '':
        segment1_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(data_model_tensor_modelName_disc, x) * percent[3]'''

    if item[4] != '':
        segment1_1 = jieba.lcut(item[4], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(data_model_tensor_valueName_disc, x) * percent[4]

    # # 输入的数据表表名
    # if item[1] != '':
    #     segment1_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
    #     s1 = [word_avg(model, segment1_1)]
    #     x = torch.Tensor(s1).to(device)
    #     final_value += tensor_module(torch.load(model_modelName_tensor_path), x) * percent[1]

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
        res.append(data_model[i[0]])
    # print('计算出的匹配值：')
    for i in sim_value:
        # print(i[0])
        if i[0] > 1:
            i[0] = 1.0
        res_sim_value.append(i[0])
    return res, res_sim_value


def load_model_data():
    global data_model
    model_df = pd.read_csv(exec_model_path, encoding='utf-8')
    data_model = [str(x[0]) for x in model_df.values]


def save_data(demand_data, k):
    sim_words = {}

    # item1(输入): 部门-表名称-表描述-字段名称-字段描述
    # item2(加入内存的数据格式): 模型表名-模型字段名
    item1 = demand_data.split(' ')
    for data in data_model:
        sim = 0
        item2 = data.split(' ')
        # sim += bert_sim.predict(item1[0], item2[0])[0][1] * percent[0]
        # sim += bert_sim.predict(item1[1], item2[1])[0][1] * percent[1]
        # sim += bert_sim.predict(item1[2], item2[2])[0][1] * percent[2]
        # 数据表部门名与模型表部门名
        sim += bert_sim.predict(item1[0], item2[0])[0][1] * percent[0]
        # 数据表表名与模型表表名
        sim += bert_sim.predict(item1[1], item2[1])[0][1] * percent[1]
        # 数据表字段与模型属性
        sim += bert_sim.predict(item1[3], item2[3])[0][1] * percent[3]

        sim += bert_sim.predict(item1[2], item2[2])[0][1] * percent[2]
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
        res.append(sim_word[1])
    bert_data[demand_data] = res


def save_result(temp, res, query_id, sim_value):
    # single_res = []
    # for i in range(len(temp)):
    #     d = temp[i]
    #     tmp = d.split(' ')
    #     single_res.append({'originalCode': 'data2model', 'originalData': {'modelName':tmp[0],
    #                                                                       'itemName':tmp[1]},
    #                        'similarity': sim_value[i]})
    # res['key'] = query_id
    # res['result'] = single_res
    # return res

    single_res = []
    for i in range(len(temp)):
        d = temp[i]
        tmp = d.split(' ')

        single_res.append({'str': ' '.join(tmp[:5]),
                           'originalCode': tmp[5],
                           'originalData': tmp[6],
                           'similarity': sim_value[i]})

    res['key'] = query_id
    res['result'] = single_res
    return res
