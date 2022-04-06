# coding=utf-8

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
import gensim
import jieba
import torch
import xlrd
import pandas as pd
import configparser
from similarity.tools import root_path
from rest_framework.response import Response
from similarity.tools import model_dir, data_dir
from similarity.word2vec_similarity_catalog import find_data, word_avg
from similarity.word2vec_similarity_catalog import device, model, delete_ndarray, \
    bert_sim, executor, tensor_module, model_path
from similarity.database_get import db

model_data_path = os.path.join(data_dir, '01人口源表.xlsx')
# 处理完后的目录表路径
exec_data_path = os.path.join(data_dir, 'model_data.csv')
model_data = []
model_data_number = 7500
model_data_vector_department = []
model_data_vector_table = []
model_data_vector_item = []
model_data_tensor_department = None
model_data_tensor_table = None
model_data_tensor_item = None
# 数据表机构名tensor的保存路径
data_department_tensor_path = os.path.join(data_dir, 'data_department.pt')
# 数据表表名tensor的保存路径
data_table_tensor_path = os.path.join(data_dir, 'data_table.pt')
# 数据表字段tensor的保存路径
data_item_tensor_path = os.path.join(data_dir, 'data_item.pt')
bert_data = {}
query_data = {}
percent = [0.4, 0.1, 0, 0.3, 0]

keyword = 'common_data'
read_ini = configparser.ConfigParser()
read_ini.read(os.path.join(root_path,'config.ini'), encoding='utf-8')

data_col = [int(x) for x in read_ini.get(keyword, 'data_col').split(',')]
table_name = read_ini.get(keyword, 'table_name')
business_type = 'model_data'


def init_model_vector_data(request):
    global process
    global model_data_path
    global model_data
    global model_data_number
    global model_data_vector_department
    global model_data_vector_table
    global model_data_vector_item
    global model_data_tensor_department
    global model_data_tensor_table
    global model_data_tensor_item

    parameter = request.data
    # # 目录表路径
    # model_data_path = parameter['filePath']
    # if not os.path.exists(model_data_path):
    #     return Response({"code": 404, "msg": "数据表路径不存在", "data": ""})
    process = 0
    # 重新加载模型
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    process = 0.5
    # 重新缓存向量
    model_data = []
    model_data_vector_department = []
    model_data_vector_table = []
    model_data_vector_item = []
    # prepare_model_data(path=model_data_path)
    prepare_model_data()
    process = 0.75
    model_data_number = len(model_data)
    for i in range(len(model_data)):
        process = 0.75 + i / (model_data_number * 4)
        data = model_data[i]
        item = data.split(' ')
        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data_vector_department.append(s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data_vector_table.append(s2)

        segment2_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data_vector_item.append(s2)
    model_data_tensor_department = torch.Tensor(model_data_vector_department).to(device)
    model_data_tensor_table = torch.Tensor(model_data_vector_table).to(device)
    model_data_tensor_item = torch.Tensor(model_data_vector_item).to(device)

    # torch.save(model_data_tensor_department, data_department_tensor_path)
    # torch.save(model_data_tensor_table, data_table_tensor_path)
    # torch.save(model_data_tensor_item, data_item_tensor_path)

    bert_data.clear()
    query_data.clear()
    return Response({"code": 200, "msg": "词模型初始化完成；词向量缓存完成！", "data": ""})

def prepare_model_data():
    global model_data
    global table_name
    # # 打开excel
    # wb = xlrd.open_workbook(path)
    # # 按工作簿定位工作表
    # sh = wb.sheet_by_name('业务层表结构分析')
    # row_number = sh.nrows
    # for i in range(1, row_number):
    #     model_data.append(sh.cell(i, 0).value + ' ' + sh.cell(i, 1).value + ' ' + sh.cell(i, 2).value)
    # # print(model_data)

    # re = db.get_colum_by_num(data_col, table_name)
    # for i in re:
    #     while (None in i):
    #         i[i.index(None)] = '*'
    #     model_data.append(' '.join(i))
    #
    #
    # data_df = pd.DataFrame(model_data)
    # data_df.to_csv(exec_data_path, encoding='utf-8_sig', index=False)

    re = db.get_data_by_type_v2(data_col, business_type, table_name)
    for i in re:
        model_data.append(' '.join([i[0].replace('-', ' '), i[1], i[2]]))

    # print('model_data：' + str(len(model_data)))
    # for i in range(len(model_data)):
    #     print(model_data[i])



def increment_business_model_data(request):
    global model_data_vector_department
    global model_data_vector_table
    global model_data_vector_item
    global model_data_tensor_department
    global model_data_tensor_table
    global model_data_tensor_item
    global model_data
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        # match_str.replace('-', ' ')
        # 加入缓存中
        # tmp = original_data['departmentName'] + ' ' + original_data['catalogName'] + ' ' + \
        #       original_data['infoItemName'] + ' ' + original_data['departmentID'] + ' ' + original_data['catalogID']

        if len(match_str.split('-')) != 5:
            return Response({"code": 200, "msg": "新增数据失败，有效数据字段不等于5", "data": ""})

        tmp = ' '.join(match_str.split('-'))
        tmp += (' ' + original_code + ' ' + original_data)
        model_data.append(tmp)

        # print('增加后：')
        # print('model_data：' + str(len(model_data)))
        # for i in range(len(model_data)):
        #     print(model_data[i])


        item = match_str.split('-')
        # model_data.append(item[0] + ' ' + item[1] + ' ' + item[3])
        model_data.append(tmp)


        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data_vector_department.append(s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data_vector_table.append(s2)

        segment2_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        model_data_vector_item.append(s2)
    model_data_tensor_department = torch.Tensor(model_data_vector_department).to(device)
    model_data_tensor_table = torch.Tensor(model_data_vector_table).to(device)
    model_data_tensor_item = torch.Tensor(model_data_vector_item).to(device)
    bert_data.clear()
    query_data.clear()
    return Response({"code": 200, "msg": "新增数据成功！", "data": ""})

def delete_business_model_data(request):
    global model_data_tensor_department
    global model_data_tensor_table
    global model_data_tensor_item
    global model_data_vector_department
    global model_data_vector_table
    global model_data_vector_item
    global model_data
    parameter = request.data
    full_data = parameter['data']
    for single_data in full_data:
        match_str = single_data['matchStr']
        original_code = single_data['originalCode']
        original_data = single_data['originalData']
        # 加入缓存中
        # tmp = original_data['departmentName'] + ' ' + original_data['catalogName'] + ' ' + \
        #       original_data['infoItemName'] + ' ' + original_data['departmentID'] + ' ' + original_data['catalogID']

        tmp = ' '.join(match_str.split('-'))
        tmp += (' ' + original_code + ' ' + original_data)
        # print('待删除数据：')
        # print(tmp)


        # 在目录列表中删除数据
        try:
            model_data.remove(tmp)
        except:
            return Response({"code": 200, "msg": "无该数据！", "data": ""})


        # print('删除后：')
        # print('model_data：' + str(len(model_data)))
        # for i in range(len(model_data)):
        #     print(model_data[i])

        item = match_str.split('-')

        segment2_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(model_data_vector_department, s2)

        segment2_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(model_data_vector_table, s2)

        segment2_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
        s2 = word_avg(model, segment2_1)
        delete_ndarray(model_data_vector_item, s2)
    model_data_tensor_department = torch.Tensor(model_data_vector_department).to(device)
    model_data_tensor_table = torch.Tensor(model_data_vector_table).to(device)
    model_data_tensor_item = torch.Tensor(model_data_vector_item).to(device)
    bert_data.clear()
    query_data.clear()
    return Response({"code": 200, "msg": "删除数据成功！", "data": ""})

def model2data_recommend(request):
    global model_data
    global percent
    parameter = request.data
    full_data = parameter['data']
    k = parameter['k']
    weight_percent = parameter['percent']
    if len(weight_percent.split(',')) != 5:
        return Response({"code": 404, "msg": "权重配置错误！", "data": ''})
    percent = [float(x) for x in weight_percent.split(',')]
    # load_data_data()
    if len(model_data) == 0:
        return Response({"code": 404, "msg": "模型和向量未初始化！", "data": ''})
    # 顺序是部门-模型名称-模型描述-属性名称-属性描述
    source_data = []
    for i in range(len(full_data)):
        source_data.append(full_data[i]['matchStr'].replace('-', ' '))
    result = []
    # print('匹配')
    # print(len(model_data))
    # print('-' * 50)
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

        # 缓存中不存在, 后台线程缓存
        executor.submit(save_data, data, k)

    return Response({"code": 200, "msg": "查询成功！", "data": result})

def string_matching(demand_data, k):
    # res = []
    # for data in model_data:
    #     tmp_data = data.split(' ')
    #     tmp_demand_data = demand_data.split(' ')
    #     if tmp_demand_data[0] + ' ' + tmp_demand_data[1] + ' ' + tmp_demand_data[3] == data:
    #         res.append(data)
    #         if len(res) == k:
    #             break
    # return res

    res = []
    # print('data_len：' + str(len(model_data)))

    # for i in range(len(model_data)):
    #     print(model_data[i])

    for data in model_data:
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

    # 输入的模型表的字段名
    segment1_1 = jieba.lcut(item[3], cut_all=True, HMM=True)
    s1 = [word_avg(model, segment1_1)]
    x = torch.Tensor(s1).to(device)
    # final_value = tensor_module(torch.load(data_item_tensor_path), x) * percent[3]
    final_value = tensor_module(model_data_tensor_item, x) * percent[3]

    # 输入的模型表的部门
    if item[0] != '':
        segment1_1 = jieba.lcut(item[0], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        # final_value += tensor_module(torch.load(data_department_tensor_path), x) * percent[0]
        final_value += tensor_module(model_data_tensor_department, x) * percent[0]

    # 输入的模型表的部门
    if item[1] != '':
        segment1_1 = jieba.lcut(item[1], cut_all=True, HMM=True)
        s1 = [word_avg(model, segment1_1)]
        x = torch.Tensor(s1).to(device)
        final_value += tensor_module(model_data_tensor_table, x) * percent[1]

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
        res.append(model_data[i[0]])
    # print('计算出的匹配值：')
    for i in sim_value:
        # print(i[0])
        if i[0] > 1:
            i[0] = 1.0
        res_sim_value.append(i[0])
    return res, res_sim_value

def load_data_data():
    global model_data
    data_df = pd.read_csv(exec_data_path, encoding='utf-8')
    model_data = [str(x[0]) for x in data_df.values]


def save_data(demand_data, k):
    sim_words = {}

    # item1(输入): 部门-模型名称-模型描述-属性名称-属性描述
    # item2(加入内存的数据格式): 机构名-数据表名-数据表字段名
    item1 = demand_data.split(' ')
    for data in model_data:
        sim = 0
        item2 = data.split(' ')
        # sim += bert_sim.predict(item1[0], item2[0])[0][1] * percent[0]
        # sim += bert_sim.predict(item1[1], item2[1])[0][1] * percent[1]
        # sim += bert_sim.predict(item1[2], item2[2])[0][1] * percent[2]
        # 部门名与机构名
        sim += bert_sim.predict(item1[0], item2[0])[0][1] * percent[0]
        # 模型表名与数据表名
        sim += bert_sim.predict(item1[1], item2[1])[0][1] * percent[1]
        # 模型属性名与数据字段
        sim += bert_sim.predict(item1[3], item2[3])[0][1] * percent[3]

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
    #     single_res.append({'originalCode': 'model2data', 'originalData': {'instituteName':tmp[0],
    #                                                                       'tableName':tmp[1],
    #                                                                       'itemName':tmp[2]},
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