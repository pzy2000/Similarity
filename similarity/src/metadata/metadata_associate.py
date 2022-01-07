# -*- coding:utf-8 -*-
import os
# import django
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

import warnings
import pandas as pd
# import tensorflow as tf
import numpy as np
import random
from difflib import SequenceMatcher
import json
import operator
from tqdm import tqdm
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
# from similarity.src.BERT.run_similarity import BertSim
from similarity.src.metadata.metadata_config import Config, data_path, result_path




warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'



class MetaData(object):

    def __init__(self, config):
        self.config = config
        self.metadata_load_path = self.config.metadata_path
        self.metadata_save_path = os.path.join(data_path, 'metadata.csv')
        self.model_path = self.config.model_path
        self.catalogue_path = self.config.catalogue_path
        self.exist_asso_path = self.config.exist_asso_path
        self.top_k = self.config.top_k
        self.metadata_df = None     # 数据元dataframe
        self.model_df = None        # 模型表dataframe
        self.catalogue_df = None    # 目录表信息项dataframe
        self.exist_asso_df = None   # 已存在关联关系的dataframe
        self.train_df = None        # 训练集dataframe
        self.test_df = None         # 测试集dataframe
        self.asso_meta_model = {}           # 数据元与模型表的关联关系{meta1:[[ins1,table1,word1],[ins2,table2,word2]]}
        self.asso_model_meta = {}           # 模型表与数据元的关联关系{index1:[[ins1,table1,word1],meta1],index2:[[ins2,table2,word2],meta2]}
        self.asso_model_multimeta = {}      # 模型表与数据元的多关联关系{index1:[[ins1,table1,word1],[meta1,meta22,meta35]],index2:[[ins2,table2,word2],[meta12,meta20,meta13]]}
        self.asso_meta_catalogue = {}       # 数据元与目录表信息项的关联关系{meta1:[[catalogue_name1,word1,ins1],[catalogue_name2,word2,ins2]]}
        self.asso_catalogue_meta = {}       # 目录表信息项与数据元的关联关系{index1:[[catalogue_name1,word1,ins1],meta1],index2:[[catalogue_name2,word2,ins2],meta2]}
        self.asso_catalogue_multimeta = {}  # 目录表信息项与数据元的多关联关系{index1:[[catalogue_name1,word1,ins1],[meta1,meta22,meta35]],index2:[[catalogue_name2,word2,ins2],[meta12,meta20,meta13]]}
        self.exist_asso_dic = {}            # 已存在的关联关系
        # self.bert_sim = BertSim()
        # self.bert_sim.set_mode(tf.estimator.ModeKeys.PREDICT)

    def load_metadata(self):
        '''
        加载数据元
        :return:
        '''
        self.metadata_df = pd.read_csv(self.metadata_load_path, encoding='utf-8')
        # 去除重复行
        self.metadata_df = self.metadata_df.drop_duplicates()
        print('数据元数据量：' + str(len(self.metadata_df)))
        print('-'*25+'数据元加载完成'+'-'*25)

    def add_metadata(self):
        if self.config.add_metadata_path != None and \
           self.config.add_metadata_sheet != None and \
           self.config.add_metadata_col != None:
            # 当原始数据元路径不存在时，则将新增的数据元作为原始数据元处理
            if os.path.exists(self.config.metadata_path) is False:
                add_df = pd.read_excel(self.config.add_metadata_path, encoding='utf-8',
                                       sheet_name=self.config.add_metadata_sheet)
                # 新增的数据元去空去重操作
                add_df = add_df[self.config.add_metadata_col].dropna()
                add_df = add_df.drop_duplicates()
                add_df.to_csv(self.config.metadata_path, encoding='utf-8_sig', index=False, header=['数据元'])
                print('-' * 25 + '原始数据元添加成功' + '-' * 25)
            else:
                # 原来已存在数据元
                origin_metadata_df = pd.read_csv(self.config.metadata_path, encoding='utf-8')
                add_df = pd.read_excel(self.config.add_metadata_path, encoding='utf-8', sheet_name=self.config.add_metadata_sheet)
                # 新增的数据元去空去重操作
                add_df = add_df[self.config.add_metadata_col].dropna()
                add_df = add_df.drop_duplicates()
                # 与原来存在的数据元进行判重操作
                total_df = origin_metadata_df['数据元'].append(add_df)
                total_df = total_df.drop_duplicates()
                total_df.to_csv(self.config.metadata_path, encoding='utf-8_sig', index=False, header=['数据元'])
                print('-' * 25 + '新增数据元添加成功' + '-' * 25)
        else:
            print('路径、表单名或列名为空，新增失败！')

    def load_model(self):
        '''
        加载模型表数据
        :return:
        '''
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
        with open(os.path.join(data_path, 'meta_dic.json'), 'w', encoding='utf-8') as f:
            json.dump(self.exist_asso_dic, f, ensure_ascii=False, indent=2)
        # 随机采样产生训练集与测试集
        print('现存关联数据量：' + str(len(self.exist_asso_df)))
        print('-'*25+'已存在关联数据加载完成'+'-'*25)

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
        # if isBert == True:
        #     if sim <= 0.1:
        #         sim = self.bert_sim.predict(metadata, info)[0][1]
        return sim

    def model_preprocess(self):
        '''
        模型表与数据元关联前的预处理，包括加载数据元，加载新增数据元，加载模型表，加载已存在的关联关系
        :return:
        '''
        self.add_metadata()
        self.load_metadata()
        self.load_model()
        self.load_exist_assoc()

    def catalogue_preprocess(self):
        '''
        目录表信息项与数据元关联前的预处理，包括加载数据元，加载新增数据元，加载目录表，加载已存在的关联关系
        :return:
        '''

        self.add_metadata()
        self.load_metadata()
        self.load_catalogue()
        self.load_exist_assoc()

    def model_associate(self, metadata, model,
                  meta_model_path=os.path.join(result_path, 'model_table\\multi\\meta_model_multi.json'),
                  model_meta_path=os.path.join(result_path, 'model_table\\multi\\model_meta_multi.json'),
                  model_multimeta_path=os.path.join(result_path, 'model_table\\multi\\model_meta_top5_multi.json'),
                  model_asso_path=os.path.join(result_path, 'model_table\\multi\\model_asso.txt')):

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
            for j in range(len(metadata_list)):
                # 计算每个模型表中的字段与数据元字段的相似度
                similarity = self.sim_common_str(model_list[i][2], metadata_list[j])
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
                for m in range(self.top_k-len(sim_index)):
                    tmp = [0,random.randint(0,len(metadata_list)-1)]
                    sim_index.append(tmp)

            sim_index = sorted(sim_index, key=lambda x:x[0], reverse=True)
            # print(sim_index[0][1])
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


        # 将关联关系保存为json文件
        with open(meta_model_path, 'w', encoding='utf-8') as f:
            json.dump(self.asso_meta_model, f, ensure_ascii=False, indent=2)

        with open(model_meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.asso_model_meta, f, ensure_ascii=False, indent=2)
        # 如果路径仍未默认，则并没有被就修改，按用户设置top_k格式命名
        if model_multimeta_path == os.path.join(result_path, 'model_table\\multi\\model_meta_top5_multi.json'):
            model_multimeta_path = os.path.join(result_path, 'model_table\\multi\\model_meta_top'+ str(self.top_k) + '_multi.json')
        with open(model_multimeta_path, 'w', encoding='utf-8') as f:
            json.dump(self.asso_model_multimeta, f, ensure_ascii=False, indent=2)

        print('-' * 25 + '模型表与数据元字段自动关联完成' + '-' * 25)

        self.model_save_asso(model_multimeta_path, model_asso_path)

    def catalogue_associate(self, metadata, catalogue, isBert=False,
                  meta_catalogue_path=os.path.join(result_path, 'catalogue_table\\multi\\meta_catalogue_multi.json'),
                  catalogue_meta_path=os.path.join(result_path, 'catalogue_table\\multi\\catalogue_meta_multi.json'),
                  catalogue_multimeta_path=os.path.join(result_path, 'catalogue_table\\multi\\catalogue_meta_top5_multi.json'),
                  catalogue_asso_path=os.path.join(result_path, 'catalogue_table\\multi\\catalogue_asso.txt')):

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
            for j in range(len(metadata_list)):
                # 计算每个目录表信息项中的字段与数据元字段的相似度
                similarity = 0
                similarity = self.sim_common_str(catalogue_list[i][1], metadata_list[j])
                # similarity = self.catalogue_bert(catalogue_list[i][1], metadata_list[j], isBert)
                # 在现存关联关系表中是否存在，若存在则权重增加
                if (catalogue_list[i][1] in self.exist_asso_dic.keys()) and \
                    self.exist_asso_dic[catalogue_list[i][1]] == metadata_list[j]:
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
                    tmp = [0, random.randint(0, len(metadata_list)-1)]
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

        # 将关联关系保存为json文件
        with open(meta_catalogue_path, 'w', encoding='utf-8') as f:
            json.dump(self.asso_meta_catalogue, f, ensure_ascii=False, indent=2)

        with open(catalogue_meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.asso_catalogue_meta, f, ensure_ascii=False, indent=2)
        # 如果路径仍未默认，则并没有被就修改，按用户设置top_k格式命名
        if catalogue_multimeta_path == os.path.join(result_path, 'catalogue_table\\multi\\catalogue_meta_top5_multi.json'):
            catalogue_multimeta_path = os.path.join(result_path, 'catalogue_table\\multi\\catalogue_meta_top' + str(self.top_k) + '_multi.json')
        with open(catalogue_multimeta_path, 'w', encoding='utf-8') as f:
            json.dump(self.asso_catalogue_multimeta, f, ensure_ascii=False, indent=2)

        print('-' * 25 + '目录表数据项与数据元字段自动关联完成' + '-' * 25)

        self.catalogue_save_asso(catalogue_multimeta_path, catalogue_asso_path)

    def model(self):
        # 加载数据元、模型表以及已存在的关联关系
        self.model_preprocess()
        # 数据元与模型表的关联
        model_meta_path = self.config.model_save_path + self.config.model_meta_name
        meta_model_path = self.config.model_save_path + self.config.meta_model_name
        model_multimeta_path = self.config.model_save_path + self.config.model_multimeta_name
        model_asso_path = self.config.model_save_path + self.config.model_asso_name
        self.model_associate(self.metadata_df, self.model_df,model_meta_path,
                             meta_model_path,model_multimeta_path,model_asso_path)

        # 数据元与模型表关联关系的评估
        # self.model_evaluate(model_multimeta_path)
        return self.asso_model_multimeta

    def catalogue(self):
        # 加载数据元、模型表以及已存在的关联关系
        self.catalogue_preprocess()
        # 数据元与目录表信息项的关联
        catalogue_meta_path = self.config.catalogue_save_path + self.config.catalogue_meta_name
        meta_catalogue_path = self.config.catalogue_save_path + self.config.meta_catalogue_name
        catalogue_multimeta_path = self.config.catalogue_save_path + self.config.catalogue_multimeta_name
        catalogue_asso_path = self.config.catalogue_save_path + self.config.catalogue_asso_name
        self.catalogue_associate(self.metadata_df, self.catalogue_df, isBert=False,
                                    catalogue_meta_path=catalogue_meta_path,
                                    meta_catalogue_path=meta_catalogue_path,
                                    catalogue_multimeta_path=catalogue_multimeta_path,
                                    catalogue_asso_path=catalogue_asso_path)

        # 数据元与目录表信息项关联关系的评估
        # metadata.catalogue_evaluate(min_confid=self.config.min_confid, max_confid=self.config.max_confid,
        #                             json_path=catalogue_multimeta_path)

        # self.catalogue_evaluate_new(json_path=catalogue_multimeta_path)
        return self.asso_catalogue_multimeta

    def sim_common_str(self, test_data, metadata):
        '''
        使用公共子序列比较两字符串是否相同
        :param test_data: 表中字段
        :param metadata: 数据元字段
        :return: 返回两字符串的相似度
        '''
        # similarity = 0
        similarity = SequenceMatcher(None, test_data, metadata).ratio()

        return similarity

    def model_save_asso(self, json_path, save_path):
        exist_asso_list = self.exist_asso_df.values.tolist()
        model_list = self.model_df.values.tolist()
        asso_txt_file = open(save_path, 'w+', encoding='utf-8')

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            # {index1:[[ins1,table1,word1],[meta1,meta22,meta35]]}
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
            # {index1:[[table1,word1,ins1],[meta1,meta22,meta35]]}
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
        for i in range(len(exist_asso_list)):
            if operator.eq(model_list, exist_asso_list[i][:3]) is True:
                index = i
                break
        return index

    def model_evaluate(self, json_path=os.path.join(result_path, 'model_table\\multi\\model_meta_multi.json')):
        exist_asso_list = self.exist_asso_df.values.tolist()
        model_list = self.model_df.values.tolist()

        # top_1数据元的命中个数
        count = 0
        # 前k个数据元中命中个数
        top_k_count = 0
        # 加载模型表与数据元关联json文件
        # {index1:[[ins1,table1,word1],meta1],index2:[[ins2,table2,word2],meta2]}
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for key, value in json_data.items():
            index = int(key)
            # 若模型表中下标内容与现存关联关系表中同位置内容不一致时对下标进行重映射
            if(operator.eq(model_list[index], exist_asso_list[index][:3]) is False):
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
                print('-'*50)
        print('top_1：' + str(count))
        print('top_' + str(self.top_k) + '：' + str(top_k_count))
        print('acc：', count / len(self.model_df))
        print('top_' + str(self.top_k) + '_acc：', top_k_count / len(self.model_df))

    def catalogue_evaluate(self, min_confid=0.3, max_confid=0.7,
                           json_path=os.path.join(result_path, 'catalogue\\multi\\catalogue_meta_multi.json')):
        catalogue_list = self.catalogue_df.values.tolist()
        catalogue_len = len(self.catalogue_df)

        # top_1数据元的命中个数
        count = 0
        # 前k个数据元中命中个数
        top_k_count = 0
        # 加载目录表信息项与数据元关联json文件
        # {index1:[[catalogue_name1,word1,ins1],meta1],index2:[[catalogue_name2,word2,ins2],meta2]}
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for key, value in json_data.items():
            index = int(key)
            # 前者为真实值，后者为预测值
            if catalogue_list[index][1] in value[1]:
                top_k_count += 1
            else: # 当候选数据元中最大匹配度大于置信度时则认为已被匹配
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

    def catalogue_evaluate_new(self, json_path=os.path.join(result_path, 'catalogue\\multi\\catalogue_meta_multi.json')):

        # 现存目录表信息项与数据元的关联关系
        catalogue_exist_asso_df = pd.read_excel(os.path.join(data_path, '某区目录-信息项数据2.0.xlsx'), encoding='utf-8', sheet_name='某区目录信息项')
        catalogue_exist_asso_df = catalogue_exist_asso_df.iloc[:,[0,1,4,2]]
        catalogue_exist_asso_list = catalogue_exist_asso_df.values.tolist()

        # 纯目录表信息项
        catalogue_list = self.catalogue_df.values.tolist()
        catalogue_len = len(self.catalogue_df)
        origin_catalogue_len = catalogue_len

        # top_1数据元的命中个数
        count = 0
        # 前k个数据元中命中个数
        top_k_count = 0
        # 加载目录表信息项与数据元关联json文件
        # {index1:[[catalogue_name1,word1,ins1],meta1],index2:[[catalogue_name2,word2,ins2],meta2]}
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


config = Config()
init_flag = False

@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def init_data_path(request):
    global init_flag
    global config
    parameter = request.data
    # 数据元路径
    config.metadata_path = parameter['metadata_path']
    # 新增数据元路径
    config.add_metadata_path = parameter['add_metadata_path']
    # 新增数据元表单名称
    config.add_metadata_sheet = parameter['add_metadata_sheet']
    # 新增数据元表列名称
    config.add_metadata_col = parameter['add_metadata_col']
    # 模型表路径
    config.model_path = parameter['model_path']
    # 目录表路径
    config.catalogue_path = parameter['catalogue_path']
    # 前k个候选数据元
    config.top_k = parameter['top_k']
    # 模型表关联结果保存路径
    config.model_save_path = parameter['model_save_path']
    # 目录表关联结果保存路径
    config.catalogue_save_path = parameter['catalogue_save_path']
    init_flag = True
    return HttpResponse("文件路径初始化成功")



@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def single_match(request):
    global init_flag
    if init_flag == False:
        return HttpResponse("文件路径未初始化")

    metadata = MetaData(config)
    global re
    parameter = request.data
    type = parameter['type']
    if type == 'model':     # 模型表与数据元的关联
        re = metadata.model()
    elif type == 'catalogue':   # 目录表信息项与数据元的关联
        re = metadata.catalogue()
    else:
        return HttpResponse("类型错误")
    return Response(re)



if __name__ == '__main__':

    # 初始化配置
    metadata = MetaData(config)
    # ---------------------------模型表与数据元字段自动关联--------------------------
    # metadata.model()

    # ---------------------------目录表信息项与数据元字段自动关联--------------------------
    metadata.catalogue()
    # metadata.catalogue_evaluate_new()

