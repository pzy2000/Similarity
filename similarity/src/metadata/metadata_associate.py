# -*- coding:utf-8 -*-
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

import warnings
import jieba
import pandas as pd
import numpy as np
import torch
import random
from difflib import SequenceMatcher
import json
import operator
from tqdm import tqdm
from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from similarity.word2vec_similarity_catalog import model, word_avg, tensor_module, bert_sim, executor, device
from similarity.src.metadata.metadata_config import Config, data_dir, result_dir

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

metadata_tensor = None
metadata_vector = []

bert_data = {}
query_data = {}


class MetaData(object):

    def __init__(self, config):
        self.config = config
        self.metadata_load_path = self.config.metadata_path
        self.metadata_save_path = os.path.join(data_dir, 'metadata.csv')
        self.model_path = self.config.model_path
        self.catalogue_path = self.config.catalogue_path
        self.exist_asso_path = self.config.exist_asso_path
        self.top_k = self.config.top_k
        self.metadata_df = None  # 数据元dataframe
        self.model_df = None  # 模型表dataframe
        self.catalogue_df = None  # 目录表信息项dataframe
        self.exist_asso_df = None  # 已存在关联关系的dataframe
        self.train_df = None  # 训练集dataframe
        self.test_df = None  # 测试集dataframe
        self.asso_meta_model = {}  # 数据元与模型表的关联关系{meta1:[[ins1,table1,word1],[ins2,table2,word2]]}
        self.asso_model_meta = {}  # 模型表与数据元的关联关系{index1:[[ins1,table1,word1],meta1],index2:[[ins2,table2,word2],meta2]}
        self.asso_model_multimeta = {}  # 模型表与数据元的多关联关系{index1:[[ins1,table1,word1],[meta1,meta22,meta35]],index2:[[ins2,table2,word2],[meta12,meta20,meta13]]}
        self.asso_meta_catalogue = {}  # 数据元与目录表信息项的关联关系{meta1:[[catalogue_name1,word1,ins1],[catalogue_name2,word2,ins2]]}
        self.asso_catalogue_meta = {}  # 目录表信息项与数据元的关联关系{index1:[[catalogue_name1,word1,ins1],meta1],index2:[[catalogue_name2,word2,ins2],meta2]}
        self.asso_catalogue_multimeta = {}  # 目录表信息项与数据元的多关联关系{index1:[[catalogue_name1,word1,ins1],[meta1,meta22,meta35]],index2:[[catalogue_name2,word2,ins2],[meta12,meta20,meta13]]}
        self.exist_asso_dic = {}  # 已存在的关联关系

    def load_metadata(self):
        '''
        加载数据元
        :return:
        '''
        self.metadata_df = pd.read_csv(self.metadata_load_path, encoding='utf-8')
        # 去除重复行
        self.metadata_df = self.metadata_df.drop_duplicates()
        # 对字段进行词向量预处理
        global metadata_tensor
        global metadata_vector
        metadata_tensor = None
        metadata_vector = []
        metadata_list = self.metadata_df.values.tolist()
        # print(metadata_list)
        for metadata in metadata_list:
            segment2_1 = jieba.lcut(metadata[0], cut_all=True, HMM=True)
            s2 = word_avg(model, segment2_1)
            metadata_vector.append(s2)
        metadata_tensor = torch.Tensor(metadata_vector).to(device)
        print('数据元数据量：' + str(len(self.metadata_df)))
        print('-' * 25 + '数据元加载完成' + '-' * 25)

    def add_metadata(self):
        if self.config.add_metadata_path != None:
            # 原来已存在数据元
            origin_metadata_df = pd.read_csv(self.config.metadata_path, encoding='utf-8')
            add_df = pd.read_csv(self.config.add_metadata_path, encoding='utf-8')
            # 新增的数据元去空去重操作
            add_df = add_df.dropna()
            add_df = add_df.drop_duplicates()
            # 与原来存在的数据元进行判重操作
            total_df = origin_metadata_df.append(add_df, ignore_index=True)
            total_df = total_df.drop_duplicates()
            total_df.to_csv(config.metadata_path, encoding='utf-8_sig', index=False, header='数据元')
            print('-' * 25 + '新增数据元添加成功' + '-' * 25)
        else:
            print('路径、表单名或列名为空，新增失败！')

    def load_model(self):
        '''
        加载模型表数据
        :return:
        '''
        # 加载模型表数据
        self.model_df = pd.read_csv(self.model_path, encoding='utf-8')
        print('模型表数据量：' + str(len(self.model_df)))
        print('-' * 25 + '模型表数据加载完成' + '-' * 25)

    def load_catalogue(self):
        '''
        加载目录表信息项数据
        :return:
        '''
        self.catalogue_df = pd.read_csv(self.catalogue_path, encoding='utf-8')

        print('信息表信息项数据量：' + str(len(self.catalogue_df)))
        print('-' * 25 + '目录表信息项数据加载完成' + '-' * 25)

    def load_exist_assoc(self, sheet_name='模型表原始字段与数据元映射'):
        '''
        加载已存在的关联关系
        :return:
        '''
        self.exist_asso_df = pd.read_excel(self.exist_asso_path, encoding='utf-8', sheet_name=sheet_name)

        # 提取并保存现存关联关系
        exist_asso_list = self.exist_asso_df.values.tolist()
        for i in exist_asso_list:
            self.exist_asso_dic[i[1]] = i[3]
        with open(os.path.join(data_dir, 'meta_dic.json'), 'w', encoding='utf-8') as f:
            json.dump(self.exist_asso_dic, f, ensure_ascii=False, indent=2)
        # 随机采样产生训练集与测试集
        print('现存关联数据量：' + str(len(self.exist_asso_df)))
        print('-' * 25 + '已存在关联数据加载完成' + '-' * 25)

    def model_addition_info(self, metadata, ins_info, table_info, scale=0.5):
        '''
        在模型表中计算除字段外其他信息的权重
        :param metadata: 数据元
        :param ins_info: 机构信息
        :param table_info: 模型表信息
        :param scale: 调整权重
        :return: 返回根据模型权重
        '''
        sim = 0
        if type(ins_info) == type('123'):
            sim += SequenceMatcher(None, ins_info, metadata).ratio()
        if type(table_info) == type('123'):
            sim += SequenceMatcher(None, table_info, metadata).ratio()
        sim *= scale
        return sim

    def catalogue_bert(self, info, metadata, isBert=False):
        '''
        使用bert计算数据元与信息项的语义相似度
        :param info: 目录表信息项
        :param metadata: 数据元
        :param isBert: 是否采用bert的策略
        :return: 返回相似度
        '''
        sim = 0
        sim = self.sim_common_str(info, metadata)
        return sim

    def model_preprocess(self):
        '''
        模型表与数据元关联前的预处理，包括加载数据元，加载新增数据元，加载模型表，加载已存在的关联关系
        :return:
        '''
        self.load_metadata()
        self.load_model()
        self.load_exist_assoc()

    def catalogue_preprocess(self):
        '''
        目录表信息项与数据元关联前的预处理，包括加载数据元，加载新增数据元，加载目录表，加载已存在的关联关系
        :return:
        '''

        self.load_metadata()
        self.load_catalogue()
        self.load_exist_assoc()

    def model_associate(self, metadata, model,
                        meta_model_path=os.path.join(result_dir, 'model_table\\multi\\meta_model_multi.json'),
                        model_meta_path=os.path.join(result_dir, 'model_table\\multi\\model_meta_multi.json'),
                        model_multimeta_path=os.path.join(result_dir, 'model_table\\multi\\model_meta_top5_multi.json'),
                        model_asso_path=os.path.join(result_dir, 'model_table\\multi\\model_asso.txt')):

        metadata_list = []
        model_list = [[]]

        if type(metadata) is pd.DataFrame:
            metadata_list = metadata.values.tolist()
            metadata_list = list(np.array(metadata_list).flat)
        elif type(metadata) is type(metadata_list):
            metadata_list = metadata

        if type(model) is pd.DataFrame:
            model_list = model.values.tolist()
        elif type(model) is type(model_list):
            model_list = model

        for i in tqdm(range(len(model_list))):
            # 相似度与数据元下标映射表
            sim_index = []
            for j, item in enumerate(metadata_list):
                # 计算每个模型表中的字段与数据元字段的相似度
                similarity = self.sim_common_str(model_list[i][2], item)
                # 将其他的信息纳入考虑，计算相似度
                # addition_sim = self.model_addition_info(metadata_list[j], model_list[i][0], model_list[i][1])
                # print('额外信息：' + str(addition_sim))
                # if addition_sim > 0.1:
                #     similarity += addition_sim

                if similarity <= 0:
                    continue
                # 构造临时存储结构sim_index：[[sim1,meta_index1],[sim2,meta_index2]]
                tmp = []
                tmp.append(similarity)
                tmp.append(j)
                sim_index.append(tmp)
            # 当筛选出的数据元不足待选数量时进行补全
            if len(sim_index) < self.top_k:
                for m in range(self.top_k - len(sim_index)):
                    tmp = [0, random.randint(0, len(metadata_list) - 1)]
                    sim_index.append(tmp)

            sim_index = sorted(sim_index, key=lambda x: x[0], reverse=True)
            max_sim_metadata = metadata_list[sim_index[0][1]]

            # 记录每个数据元对应的表中字段,完成数据元到模型表的映射{meta1:[[ins1,table1,word1],[ins2,table2,word2]]}
            self.asso_meta_model.setdefault(max_sim_metadata, [])
            self.asso_meta_model[max_sim_metadata].append(model_list[i][:3])
            # 完成模型表到数据元的映射,index为与数据元匹配的模型表字段的下标，避免模型表字段重复的情况
            # {index1:[[ins1,table1,word1],meta1],index2:[[ins2,table2,word2],meta2]}
            top_one_metadata_list = [model_list[i][:3]]
            top_one_metadata_list.append(max_sim_metadata)
            self.asso_model_meta[i] = top_one_metadata_list

            top_k_metadata_list = [model_list[i][:3]]
            top_k_list = []
            for k in range(self.top_k):
                top_k_list.append(metadata_list[sim_index[k][1]])

            top_k_metadata_list.append(top_k_list)
            self.asso_model_multimeta[i] = top_k_metadata_list

        # # 将关联关系保存为json文件

    def catalogue_associate(self, metadata, catalogue, isBert=False,
                            meta_catalogue_path=os.path.join(result_dir,'catalogue_table\\multi\\meta_catalogue_multi.json'),
                            catalogue_meta_path=os.path.join(result_dir,'catalogue_table\\multi\\catalogue_meta_multi.json'),
                            catalogue_multimeta_path=os.path.join(result_dir,'catalogue_table\\multi\\catalogue_meta_top5_multi.json'),
                            catalogue_asso_path=os.path.join(result_dir, 'catalogue_table\\multi\\catalogue_asso.txt')):

        metadata_list = []
        catalogue_list = [[]]

        if type(metadata) is pd.DataFrame:
            metadata_list = metadata.values.tolist()
            metadata_list = list(np.array(metadata_list).flat)
        elif type(metadata) is type(metadata_list):
            metadata_list = metadata

        if type(catalogue) is pd.DataFrame:
            catalogue_list = catalogue.values.tolist()
        elif type(catalogue) is type(catalogue_list):
            catalogue_list = catalogue

        # print(catalogue)

        for i in tqdm(range(len(catalogue_list))):
            # 相似度与数据元下标映射表
            sim_index = []
            for j, item in enumerate(metadata_list):
                # 计算每个目录表信息项中的字段与数据元字段的相似度
                similarity = 0
                similarity = self.sim_common_str(catalogue_list[i][1], item)
                # similarity = self.catalogue_bert(catalogue_list[i][1], metadata_list[j], isBert)
                # 在现存关联关系表中是否存在，若存在则权重增加
                if (catalogue_list[i][1] in self.exist_asso_dic) and \
                        self.exist_asso_dic[catalogue_list[i][1]] == item:
                    similarity += 1

                if similarity <= 0:
                    continue
                # 构造临时存储结构sim_index：[[sim1,meta_index1],[sim2,meta_index2]]
                tmp = []
                tmp.append(similarity)
                tmp.append(j)

                sim_index.append(tmp)
            # 当筛选出的数据元不足待选数量时进行补全
            if len(sim_index) < self.top_k:
                for m in range(self.top_k - len(sim_index)):
                    tmp = [0, random.randint(0, len(metadata_list) - 1)]
                    sim_index.append(tmp)

            sim_index = sorted(sim_index, key=lambda x: x[0], reverse=True)
            max_sim_metadata = metadata_list[sim_index[0][1]]
            # 记录每个数据元对应的表中字段,完成数据元到目录表信息项的映射{meta1:[[catalog_name1, word1, ins1],[catalog_name2, word2, ins2]]}
            self.asso_meta_catalogue.setdefault(max_sim_metadata, [])
            self.asso_meta_catalogue[max_sim_metadata].append(catalogue_list[i][:3])
            # 完成目录表到数据元的映射,index为与数据元匹配的目录表信息项的下标，避免目录表信息项重复的情况
            # {index1:[[catalogue_name1,word1,ins1],meta1],index2:[[catalogue_name2,word2,ins2],meta2]}
            top_one_metadata_list = [catalogue_list[i][:3]]
            top_one_metadata_list.append(max_sim_metadata)
            self.asso_catalogue_meta[i] = top_one_metadata_list

            top_k_metadata_list = [catalogue_list[i][:3]]
            top_k_list = []
            for k in range(self.top_k):
                top_k_list.append(metadata_list[sim_index[k][1]])

            top_k_metadata_list.append(top_k_list)
            self.asso_catalogue_multimeta[i] = top_k_metadata_list

        # # 将关联关系保存为json文件

    def model(self):
        # 加载数据元、模型表以及已存在的关联关系
        self.model_preprocess()
        # 数据元与模型表的关联
        model_meta_path = self.config.model_save_path + self.config.model_meta_name
        meta_model_path = self.config.model_save_path + self.config.meta_model_name
        model_multimeta_path = self.config.model_save_path + self.config.model_multimeta_name
        model_asso_path = self.config.model_save_path + self.config.model_asso_name

        self.integrate_sim(self.metadata_df, self.model_df, 2, self.asso_model_multimeta)

        return self.asso_model_multimeta

    def catalogue(self):
        # 加载数据元、模型表以及已存在的关联关系
        self.catalogue_preprocess()
        # 数据元与目录表信息项的关联
        catalogue_meta_path = self.config.catalogue_save_path + self.config.catalogue_meta_name
        meta_catalogue_path = self.config.catalogue_save_path + self.config.meta_catalogue_name
        catalogue_multimeta_path = self.config.catalogue_save_path + self.config.catalogue_multimeta_name
        catalogue_asso_path = self.config.catalogue_save_path + self.config.catalogue_asso_name

        self.integrate_sim(self.metadata_df, self.catalogue_df, 1, self.asso_catalogue_multimeta)



        return self.asso_catalogue_multimeta

    def build_metadata_map(self, index, item_list,
                           metadata_list, data_multimeta_dic):
        top_k_metadata_list = [item_list]
        top_k_metadata_list.append(metadata_list)
        data_multimeta_dic[index] = top_k_metadata_list

    def save_data(self, metadata_list, item_data):
        return
        sim_words = {}
        for data in metadata_list:
            sim = bert_sim.predict(data, item_data)[0][1]
            if len(sim_words) < self.top_k:
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
        bert_data[item_data] = res

    def string_matching(self, index, item_list, metadata_list, key_index,
                        res_metadata_list, item_multimeta_dic):
        '''
        事项表字段与数据元的字符匹配
        :param index: 在事项表中的下标
        :param item_list: 事项表的一维列表[*,*,item]
        :param metadata_list: 数据元一维列表
        :param key_index: 关键字段在事项表中的下标
        :param res_metadata_list: 匹配到的结果数据元一维列表
        :param item_multimeta_dic: 事项表与数据元的映射字典，格式为
                {index1:[[ins1,table1,word1],[meta1,meta22,meta35]],
                index2:[[ins2,table2,word2],[meta12,meta20,meta13]]}
        :return: 返回匹配是否成功，True表示成功
        '''
        for metadata in metadata_list:
            if item_list[key_index] == metadata:
                res_metadata_list.append(metadata)
            # 匹配到的数据元列表不为空，则说明找到最相似的，退出循环
            # 继续寻找下一个字段关联的数据元
            if len(res_metadata_list) != 0:
                # 构建模型表字段与数据元的映射
                self.build_metadata_map(index, item_list, res_metadata_list,
                                        item_multimeta_dic)
                return True
        return False

        # sim_index = []
        # for metadata in metadata_list:
        #     similarity = self.sim_common_str(item_list[key_index], metadata)
        #     if similarity <= 0:
        #         continue
        #     # 构造临时存储结构sim_index：[[metadata1, sim1],[metadata2, sim2]]
        #     tmp = []
        #     tmp.append(metadata)
        #     tmp.append(similarity)
        #     sim_index.append(tmp)
        #     # 当筛选出的数据元不足待选数量时进行补全
        #     if len(sim_index) < self.top_k:
        #         for m in range(self.top_k - len(sim_index)):
        #             tmp = [metadata_list[random.randint(0, len(metadata_list) - 1)], 0]
        #             sim_index.append(tmp)
        #
        # sim_index = sorted(sim_index, key=lambda x: x[1], reverse=True)[:3]
        # # print(sim_index)
        # for i in sim_index:
        #     # if i[1] >= 0.5:
        #     res_metadata_list.append(i[0])
        #
        # # 匹配到的数据元列表不为空，则说明找到最相似的，退出循环
        # # 继续寻找下一个字段关联的数据元
        # if len(res_metadata_list) != 0:
        #     # 构建模型表字段与数据元的映射
        #     self.build_metadata_map(index, item_list, res_metadata_list,
        #                             item_multimeta_dic)
        #     return True
        # return False

    def bert_matching(self, index, item_list, key_index,
                      res_metadata_list, item_multimeta_dic):
        '''
        事项表字段与数据元在bert缓冲中的匹配
        :param index: 在事项表中的下标
        :param item_list: 事项表的一维列表[*,*,item]
        :param key_index: 关键字段在事项表中的下标
        :param res_metadata_list: 匹配到的结果数据元一维列表
        :param item_multimeta_dic: 事项表与数据元的映射字典，格式为
                {index1:[[ins1,table1,word1],[meta1,meta22,meta35]],
                index2:[[ins2,table2,word2],[meta12,meta20,meta13]]}
        :return: 返回匹配是否成功，True表示成功
        '''
        # 查看BERT缓存
        tmp = []
        if item_list[key_index] in bert_data:
            tmp = bert_data.get(item_list[key_index])
        if len(tmp) != 0:
            res_metadata_list.append(tmp)
            # 构建模型表字段与数据元的映射
            self.build_metadata_map(index, item_list, res_metadata_list,
                                    item_multimeta_dic)
            return True  # 表示在bert缓存中找到，继续匹配下一字段
        return False

    def query_matching(self, index, item_list, key_index,
                       res_metadata_list, item_multimeta_dic):
        '''
        事项表字段与数据元在查询query缓冲中的匹配
        :param index: 在事项表中的下标
        :param item_list: 事项表的一维列表[*,*,item]
        :param key_index: 关键字段在事项表中的下标
        :param res_metadata_list: 匹配到的结果数据元一维列表
        :param item_multimeta_dic: 事项表与数据元的映射字典，格式为
                {index1:[[ins1,table1,word1],[meta1,meta22,meta35]],
                index2:[[ins2,table2,word2],[meta12,meta20,meta13]]}
        :return: 返回匹配是否成功，True表示成功
        '''
        # 查看查询缓存
        tmp = []
        if item_list[key_index] in query_data:
            tmp = query_data.get(item_list[key_index])
        if len(tmp) == self.top_k:
            # 构建模型表字段与数据元的映射
            for x in tmp:
                res_metadata_list.append(x)
            self.build_metadata_map(index, item_list, res_metadata_list,
                                    item_multimeta_dic)
            return True
        return False

    def vector_matching(self, index, item_list, metadata_list,
                        key_index, res_metadata_list, item_multimeta_dic):
        '''
        事项表字段与数据元的词向量匹配
        :param index: 在事项表中的下标
        :param item_list: 事项表的一维列表[*,*,item]
        :param metadata_list: 数据元一维列表
        :param key_index: 关键字段在事项表中的下标
        :param res_metadata_list: 匹配到的结果数据元一维列表
        :param item_multimeta_dic: 事项表与数据元的映射字典，格式为
                {index1:[[ins1,table1,word1],[meta1,meta22,meta35]],
                index2:[[ins2,table2,word2],[meta12,meta20,meta13]]}
        :return: 返回匹配出的数据元一维列表
        '''
        # 词向量匹配
        # 字符串没有匹配项，则会进行向量相似度匹配，筛选前k个
        segment1_1 = jieba.lcut(item_list[key_index], cut_all=True, HMM=True)
        item_data_vector = [word_avg(model, segment1_1)]
        item_data_tensor = torch.Tensor(item_data_vector).to(device)
        final_value = tensor_module(metadata_tensor, item_data_tensor)
        # 输出排序并输出top-k的输出
        value, _index = torch.topk(final_value, self.top_k, dim=0, largest=True, sorted=True)
        sim_index = _index.numpy().tolist()
        tmp = []
        for k in sim_index:
            tmp.append(metadata_list[k[0]])
            res_metadata_list.append(metadata_list[k[0]])
        # 构建模型表字段与数据元的映射
        self.build_metadata_map(index, item_list, res_metadata_list,
                                item_multimeta_dic)
        return tmp

    def integrate_sim(self, metadata_df, item_df, key_index, item_multimeta_dic):
        '''
        多重方法的匹配度计算，包括字符串匹配、bert缓冲查询、query缓冲查询以及词向量相似度计算
        :param metadata_df: 数据元dataframe
        :param item_df: 事项表dataframe
        :param key_index: 事项表中关键字段所在的下标
        :param item_multimeta_dic: 事项表与数据元的映射字典
        :return:
        '''
        metadata_list = metadata_df.values.tolist()
        metadata_list = list(np.array(metadata_list).flat)
        item_list = item_df.values.tolist()
        for i in tqdm(range(len(item_list))):
            res_metadata_list = []
            # 字符串匹配
            if self.string_matching(i, item_list[i], metadata_list,
                                    key_index, res_metadata_list, item_multimeta_dic) == True:
                continue



            # 查看BERT缓存
            if self.bert_matching(i, item_list[i], key_index,
                                  res_metadata_list, item_multimeta_dic) == True:
                continue
            # 查看查询缓存
            if self.query_matching(i, item_list[i], key_index,
                                   res_metadata_list, item_multimeta_dic) == True:
                continue
            # 缓存清理FIFO
            if len(bert_data.keys()) >= 10000:
                bert_data.clear()
            if len(query_data.keys()) >= 10000:
                query_data.clear()
            # 词向量匹配
            # 字符串没有匹配项，则会进行向量相似度匹配，筛选前k个
            tmp = self.vector_matching(i, item_list[i], metadata_list,
                                       key_index, res_metadata_list, item_multimeta_dic)
            # 加入查询缓存
            query_data[item_list[i][key_index]] = tmp
            # 缓存中不存在, 后台线程缓存
            executor.submit(self.save_data, metadata_list, item_list[i])

    def integrate_model_sim(self, metadata, model_data):

        metadata_list = metadata.values.tolist()
        metadata_list = list(np.array(metadata_list).flat)

        model_list = model_data.values.tolist()
        for i in tqdm(range(len(model_list))):
            res_metadata_list = []
            for j, item in enumerate(metadata_list):
                # 字符串匹配
                if model_list[i][2] == item:
                    res_metadata_list.append(item)
                # 匹配到的数据元列表不为空，则说明找到最相似的，退出循环
                # 继续寻找下一个字段关联的数据元
                if len(res_metadata_list) != 0:
                    # 构建模型表字段与数据元的映射
                    self.build_metadata_map(i, model_list[i], res_metadata_list,
                                            self.asso_model_multimeta)
                    break
                # 查看BERT缓存
                if model_list[i][2] in bert_data:
                    tmp = bert_data.get(model_list[i][2])
                    res_metadata_list.append(tmp)
                if len(res_metadata_list) != 0:
                    # 构建模型表字段与数据元的映射
                    self.build_metadata_map(i, model_list[i], res_metadata_list,
                                            self.asso_model_multimeta)
                    break

                # 查看查询缓存
                if model_list[i][2] in query_data:
                    tmp = query_data.get(model_list[i][2])
                    res_metadata_list.append(tmp)
                if len(res_metadata_list) != 0:
                    # 构建模型表字段与数据元的映射
                    self.build_metadata_map(i, model_list[i], res_metadata_list,
                                            self.asso_model_multimeta)
                    break

                # 缓存清理FIFO
                if len(bert_data.keys()) >= 10000:
                    bert_data.clear()
                if len(query_data.keys()) >= 10000:
                    query_data.clear()

                # 词向量匹配
                # 字符串没有匹配项，则会进行向量相似度匹配，筛选前k个
                segment1_1 = jieba.lcut(model_list[i][2], cut_all=True, HMM=True)
                s1 = [word_avg(model, segment1_1)]
                x = torch.Tensor(s1).to(device)
                # print(catalogue_data_tensor)
                final_value = tensor_module(metadata_tensor, x)

                # 输出排序并输出top-k的输出
                value, index = torch.topk(final_value, self.top_k, dim=0, largest=True, sorted=True)
                sim_index = index.numpy().tolist()
                tmp = []
                for k in sim_index:
                    tmp.append(metadata_list[k[0]])
                res_metadata_list.append(tmp)
                # 构建模型表字段与数据元的映射
                self.build_metadata_map(i, model_list[i], res_metadata_list,
                                        self.asso_model_multimeta)

                # 加入查询缓存
                query_data[model_list[i][2]] = tmp
                # 缓存中不存在, 后台线程缓存
                executor.submit(self.save_data, metadata_list, model_list[i])

    def sim_common_str(self, test_data, metadata):
        '''
        使用公共子序列比较两字符串是否相同
        :param test_data: 表中字段
        :param metadata: 数据元字段
        :return: 返回两字符串的相似度
        '''
        similarity = SequenceMatcher(None, test_data, metadata).ratio()

        return similarity

    def model_save_asso(self, json_path, save_path):
        exist_asso_list = self.exist_asso_df.values.tolist()
        model_list = self.model_df.values.tolist()
        asso_txt_file = open(save_path, 'w+', encoding='utf-8')

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for key, value in json_data.items():
                index = int(key)
                if (operator.eq(model_list[index], exist_asso_list[index][:3]) is False):
                    index = self.remap_index(model_list[index], exist_asso_list)
                asso_str = exist_asso_list[index][1] + '->' + \
                           exist_asso_list[index][2] + ' 关联 ' + \
                           value[1][0] + \
                           '\t\t候选数据元：' + '、'.join(value[1][1:])
                asso_txt_file.write(asso_str + '\n')

        asso_txt_file.close()
        print('-' * 25 + '写入文件成功' + '-' * 25)

    def catalogue_save_asso(self, json_path, save_path):
        catalogue_list = self.catalogue_df.values.tolist()
        asso_txt_file = open(save_path, 'w+', encoding='utf-8')

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for key, value in json_data.items():
                index = int(key)
                asso_str = str(catalogue_list[index][0]) + '->' + \
                           str(catalogue_list[index][1]) + ' 关联 ' + \
                           str(value[1][0]) + \
                           '\t\t候选数据元：' + '、'.join(value[1][1:])

                asso_txt_file.write(asso_str + '\n')

        asso_txt_file.close()
        print('-' * 25 + '写入文件成功' + '-' * 25)

    def remap_index(self, model_list, exist_asso_list):
        '''
        在已存在的关联关系中重定向模型表中的下标
        :param model_list: 模型表一维列表
        :param exist_asso_list: 关联关系二维表
        :return: 返回模型表元素在现存关联关系表中的下标
        '''
        index = 0
        for i, item in enumerate(exist_asso_list):
            if operator.eq(model_list, item[:3]) is True:
                index = i
                break
        return index

    def model_evaluate(self, json_path=os.path.join(result_dir, 'model_table\\multi\\model_meta_multi.json')):
        exist_asso_list = self.exist_asso_df.values.tolist()
        model_list = self.model_df.values.tolist()

        # top_1数据元的命中个数
        count = 0
        # 前k个数据元中命中个数
        top_k_count = 0
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for key, value in json_data.items():
            index = int(key)
            # 若模型表中下标内容与现存关联关系表中同位置内容不一致时对下标进行重映射
            if (operator.eq(model_list[index], exist_asso_list[index][:3]) is False):
                index = self.remap_index(model_list[index], exist_asso_list)
            # 前者为真实值，后者为预测值
            if exist_asso_list[index][3] in value[1]:
                top_k_count += 1

            if exist_asso_list[index][3] == value[1][0]:
                count += 1
            else:
                print('true：', end='')
                print(exist_asso_list[index][:3], end=':')
                print(exist_asso_list[index][3])

                print('predict：', end='')
                print(exist_asso_list[index][:3], end=':')
                print(value[1])
                print('-' * 50)
        print('top_1：' + str(count))
        print('top_' + str(self.top_k) + '：' + str(top_k_count))
        print('acc：', count / len(self.model_df))
        print('top_' + str(self.top_k) + '_acc：', top_k_count / len(self.model_df))

    def model_evaluate_new(self):
        exist_asso_list = self.exist_asso_df.values.tolist()
        model_list = self.model_df.values.tolist()
        count = 0
        # 前k个数据元中命中个数
        top_k_count = 0
        for key, value in self.asso_model_multimeta.items():
            index = int(key)
            # 若模型表中下标内容与现存关联关系表中同位置内容不一致时对下标进行重映射
            if (operator.eq(model_list[index], exist_asso_list[index][:3]) is False):
                index = self.remap_index(model_list[index], exist_asso_list)
            # 前者为真实值，后者为预测值
            if exist_asso_list[index][3] in value[1]:
                top_k_count += 1

            if exist_asso_list[index][3] == value[1][0]:
                count += 1
            else:
                print('true：', end='')
                print(exist_asso_list[index][:3], end=':')
                print(exist_asso_list[index][3])

                print('predict：', end='')
                print(exist_asso_list[index][:3], end=':')
                print(value[1])
                print('-' * 50)
        print('top_1：' + str(count))
        print('top_' + str(self.top_k) + '：' + str(top_k_count))
        print('acc：', count / len(self.model_df))
        print('top_' + str(self.top_k) + '_acc：', top_k_count / len(self.model_df))

    def catalogue_evaluate(self, min_confid=0.3, max_confid=0.7,
                           json_path=os.path.join(result_dir, 'catalogue\\multi\\catalogue_meta_multi.json')):
        catalogue_list = self.catalogue_df.values.tolist()
        catalogue_len = len(self.catalogue_df)

        # top_1数据元的命中个数
        count = 0
        # 前k个数据元中命中个数
        top_k_count = 0
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for key, value in json_data.items():
            index = int(key)
            # 前者为真实值，后者为预测值
            if catalogue_list[index][1] in value[1]:
                top_k_count += 1
            else:  # 当候选数据元中最大匹配度大于置信度时则认为已被匹配
                sim = self.sim_common_str(catalogue_list[index][1], value[1][0])
                # 当相似度大于最大置信度时则认为相似
                if sim >= max_confid:
                    top_k_count += 1
                # 当相似度小于最小置信度时则认为与现有数据元无关
                elif sim < min_confid:
                    catalogue_len -= 1

            if catalogue_list[index][1] == value[1][0]:
                count += 1
            else:
                sim = self.sim_common_str(catalogue_list[index][1], value[1][0])
                print('相似度为：' + str(sim))
                if sim >= max_confid:
                    count += 1
                elif sim < min_confid:
                    catalogue_len -= 1
                print('true：', end='')
                print(catalogue_list[index][:3], end=':')
                print(catalogue_list[index][1])

                print('predict：', end='')
                print(catalogue_list[index][:3], end=':')
                print(value[1])
                print('-' * 50)

        print('after eliminate：' + str(catalogue_len))

        print('top_1：' + str(count))
        print('top_' + str(self.top_k) + '：' + str(top_k_count))
        print('acc：', count / catalogue_len)
        print('top_' + str(self.top_k) + '_acc：', top_k_count / catalogue_len)

    def catalogue_evaluate_new(self, json_path=os.path.join(result_dir, 'catalogue\\multi\\catalogue_meta_multi.json')):

        # 现存目录表信息项与数据元的关联关系
        catalogue_exist_asso_df = pd.read_excel(os.path.join(data_dir, '某区目录-信息项数据2.0.xlsx'), encoding='utf-8',
                                                sheet_name='某区目录信息项')
        catalogue_exist_asso_df = catalogue_exist_asso_df.iloc[:, [0, 1, 4, 2]]
        catalogue_exist_asso_list = catalogue_exist_asso_df.values.tolist()

        # 纯目录表信息项
        catalogue_list = self.catalogue_df.values.tolist()
        catalogue_len = len(self.catalogue_df)
        origin_catalogue_len = catalogue_len

        # top_1数据元的命中个数
        count = 0
        # 前k个数据元中命中个数
        top_k_count = 0
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for key, value in json_data.items():
            index = int(key)

            # 若目录表中下标内容与现存关联关系表中同位置内容不一致时对下标进行重映射
            if (operator.eq(catalogue_list[index], catalogue_exist_asso_list[index][:3]) is False):
                index = self.remap_index(catalogue_list[index], catalogue_exist_asso_list)

            if catalogue_exist_asso_list[index][3] != catalogue_exist_asso_list[index][3]:
                catalogue_len -= 1

            # 前者为真实值，后者为预测值
            if (catalogue_exist_asso_list[index][3] == catalogue_exist_asso_list[index][3]) and \
                    catalogue_exist_asso_list[index][3] in value[1]:
                top_k_count += 1

            if (catalogue_exist_asso_list[index][3] == catalogue_exist_asso_list[index][3]) and \
                    catalogue_exist_asso_list[index][3] == value[1][0]:
                count += 1
            else:
                if catalogue_exist_asso_list[index][3] == catalogue_exist_asso_list[index][3]:
                    print('true：', end='')
                    print(catalogue_exist_asso_list[index][:3], end=':')
                    print(catalogue_exist_asso_list[index][3])

                    print('predict：', end='')
                    print(catalogue_exist_asso_list[index][:3], end=':')
                    print(value[1])
                    print('-' * 50)

        print('origin eliminate：' + str(origin_catalogue_len))
        print('after eliminate：' + str(catalogue_len))

        print('top_1：' + str(count))
        print('top_' + str(self.top_k) + '：' + str(top_k_count))
        print('acc：', count / catalogue_len)
        print('top_' + str(self.top_k) + '_acc：', top_k_count / catalogue_len)

    def catalogue_evaluate_integrate(self):
        # 现存目录表信息项与数据元的关联关系
        catalogue_exist_asso_df = pd.read_excel(os.path.join(data_dir, '某区目录-信息项数据2.0.xlsx'), encoding='utf-8',
                                                sheet_name='某区目录信息项')
        catalogue_exist_asso_df = catalogue_exist_asso_df.iloc[:, [0, 1, 4, 2]]
        catalogue_exist_asso_list = catalogue_exist_asso_df.values.tolist()


        # 纯目录表信息项
        catalogue_list = self.catalogue_df.values.tolist()
        catalogue_len = len(self.catalogue_df)
        origin_catalogue_len = catalogue_len

        # top_1数据元的命中个数
        count = 0
        # 前k个数据元中命中个数
        top_k_count = 0

        for key, value in self.asso_catalogue_multimeta.items():
            index = int(key)

            # 若目录表中下标内容与现存关联关系表中同位置内容不一致时对下标进行重映射
            if (operator.eq(catalogue_list[index], catalogue_exist_asso_list[index][:3]) is False):
                index = self.remap_index(catalogue_list[index], catalogue_exist_asso_list)

            if catalogue_exist_asso_list[index][3] != catalogue_exist_asso_list[index][3]:
                catalogue_len -= 1

            # 前者为真实值，后者为预测值
            if (catalogue_exist_asso_list[index][3] == catalogue_exist_asso_list[index][3]) and \
                    catalogue_exist_asso_list[index][3] in value[1]:
                top_k_count += 1

            if (catalogue_exist_asso_list[index][3] == catalogue_exist_asso_list[index][3]) and \
                    catalogue_exist_asso_list[index][3] == value[1][0]:
                count += 1
            else:
                if catalogue_exist_asso_list[index][3] == catalogue_exist_asso_list[index][3]:
                    print('true：', end='')
                    print(catalogue_exist_asso_list[index][:3], end=':')
                    print(catalogue_exist_asso_list[index][3])

                    print('predict：', end='')
                    print(catalogue_exist_asso_list[index][:3], end=':')
                    print(value[1])
                    print('-' * 50)

        print('origin eliminate：' + str(origin_catalogue_len))
        print('after eliminate：' + str(catalogue_len))

        print('top_1：' + str(count))
        print('top_' + str(self.top_k) + '：' + str(top_k_count))
        print('acc：', count / catalogue_len)
        print('top_' + str(self.top_k) + '_acc：', top_k_count / catalogue_len)

# TODO:词向量匹配算法
def vector_sim(str1, str2):
    segment1 = jieba.lcut(str1)
    s1 = [word_avg(model, segment1)]
    x1 = torch.Tensor(s1).to(device)
    segment2 = jieba.lcut(str2)
    s2 = [word_avg(model, segment2)]
    x2 = torch.Tensor(s2).to(device)
    sim = tensor_module(x1, x2).numpy().tolist()[0][0]
    return sim



config = Config()
init_flag = False


@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def init_data_path(request):
    global init_flag
    global config
    global metadata_tensor
    global metadata_vector
    metadata_tensor = None
    metadata_vector = []

    parameter = request.data
    # 数据元路径
    config.metadata_path = parameter['metadata_path']
    if os.path.exists(config.metadata_path) == False:
        return Response({"code": 200, "msg": "数据元路径不存在", "data": ""})
    # 模型表路径
    config.model_path = parameter['model_path']
    if os.path.exists(config.model_path) == False:
        return Response({"code": 200, "msg": "模型表路径不存在", "data": ""})
    # 目录表路径
    config.catalogue_path = parameter['catalogue_path']
    if os.path.exists(config.catalogue_path) == False:
        return Response({"code": 200, "msg": "目录表路径不存在", "data": ""})
    # 前k个候选数据元
    config.top_k = parameter['top_k']
    if type(config.top_k) == type(1.1):
        return Response({"code": 200, "msg": "候选项必须为整型", "data": ""})
    if config.top_k <= 0 or config.top_k > 10:
        return Response({"code": 200, "msg": "候选项超出取值范围[1,10]", "data": ""})

    init_flag = True
    return Response({"code": 200, "msg": "文件路径初始化成功", "data": ""})
    # return HttpResponse("文件路径初始化成功")


@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def add_metadata(request):
    global config
    global init_flag
    if init_flag == False:
        return Response({"code": 200, "msg": "文件路径未初始化", "data": ""})

    parameter = request.data
    # 新增数据元路径
    config.add_metadata_path = parameter['add_metadata_path']
    if os.path.exists(config.add_metadata_path) == False:
        return Response({"code": 200, "msg": "新增数据元路径不存在", "data": ""})
    metadata = MetaData(config)
    metadata.add_metadata()
    return Response({"code": 200, "msg": "新增数据元成功", "data": ""})
    # return HttpResponse("文件路径初始化成功")


@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def single_match(request):
    global init_flag
    if init_flag == False:
        return Response({"code": 200, "msg": "文件路径未初始化", "data": ""})

    metadata = MetaData(config)
    global re
    parameter = request.data
    type = parameter['type']
    if type == 'model':  # 模型表与数据元的关联
        re = metadata.model()
    elif type == 'catalogue':  # 目录表信息项与数据元的关联
        re = metadata.catalogue()
    else:
        return Response({"code": 200, "msg": "类型错误", "data": ""})
    # return Response(re)
    return Response({"code": 200, "msg": "关联成功", "data": re})


if __name__ == '__main__':


    # 初始化配置
    metadata = MetaData(config)

    # ---------------------------模型表与数据元字段自动关联--------------------------
    # time_start = time.time()
    metadata.model()
    # time_end = time.time()
    # print('耗时：' + str(time_end-time_start))
    # ---------------------------目录表信息项与数据元字段自动关联--------------------------
    # time_start = time.time()
    # metadata.catalogue()
    # time_end = time.time()
    # print('耗时：' + str(time_end - time_start))
    # metadata.catalogue_evaluate_new()
